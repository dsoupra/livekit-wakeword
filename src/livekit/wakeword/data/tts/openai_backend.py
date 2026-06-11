"""OpenAI-style TTS backend using /v1/audio/speech with multi-threaded requests."""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from ...config import WakeWordConfig

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


class OpenAiBackend:
    """TTS backend utilizing OpenAI-compatible API endpoints."""

    def __init__(self, config: WakeWordConfig) -> None:
        self._api_key = config.openai_tts.api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = config.openai_tts.base_url.rstrip("/")
        self._model = config.openai_tts.model
        self._voices = config.openai_tts.voices
        self._languages = config.openai_tts.languages
        self._instructions = config.openai_tts.instructions
        self._concurrency = config.openai_tts.concurrency
        self._max_retries = config.openai_tts.max_retries
        self._response_format = config.openai_tts.response_format

    @classmethod
    def from_config(cls, config: WakeWordConfig) -> OpenAiBackend:
        return cls(config)

    def validate_artifacts(self) -> None:
        if not self._base_url:
            raise ValueError("openai_tts.base_url must be configured")
        if not self._voices:
            raise ValueError("openai_tts.voices must be non-empty")

        # Log a warning if using OpenAI directly but API key is missing
        if "api.openai.com" in self._base_url and not self._api_key:
            logger.warning(
                "No API key detected for OpenAI endpoint. "
                "Ensure OPENAI_API_KEY env var is set or config has api_key."
            )

    def _request_with_retry(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> bytes:
        req_body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=req_body, headers=headers, method="POST")

        retries = 0
        backoff = 1.0
        while True:
            try:
                # 30-second timeout for TTS API requests
                with urllib.request.urlopen(req, timeout=30) as response:
                    data: bytes = response.read()
                    return data
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and retries < self._max_retries:
                    sleep_time = backoff + random.uniform(0.0, 0.5)
                    logger.warning(
                        "HTTP %d error calling %s. Retrying in %.2f seconds...",
                        e.code,
                        url,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    retries += 1
                    backoff *= 2.0
                else:
                    logger.error("HTTP error %d calling %s: %s", e.code, url, e.reason)
                    raise
            except Exception as e:
                if retries < self._max_retries:
                    sleep_time = backoff + random.uniform(0.0, 0.5)
                    logger.warning(
                        "Network error calling %s: %s. Retrying in %.2f seconds...",
                        url,
                        e,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    retries += 1
                    backoff *= 2.0
                else:
                    logger.error("Network error after %d retries: %s", self._max_retries, e)
                    raise

    def _generate_single_clip(
        self,
        sample_idx: int,
        phrase: str,
        voice: str,
        lang: str | None,
        inst: str | None,
        out_path: Path,
    ) -> Path:
        """Fetch synthetic audio from endpoint, resample, and write out_path."""
        url = f"{self._base_url}/audio/speech"
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict[str, Any] = {
            "model": self._model,
            "input": phrase,
            "voice": voice,
            "response_format": self._response_format,
        }
        if lang is not None:
            payload["language"] = lang
        if inst is not None:
            payload["instruction"] = inst

        # Get raw response audio bytes
        audio_bytes = self._request_with_retry(url, payload, headers)

        # Write to a temp file so soundfile / librosa can parse it
        with tempfile.NamedTemporaryFile(suffix=f".{self._response_format}", delete=False) as f:
            f.write(audio_bytes)
            temp_path = Path(f.name)

        try:
            # Load and resample to TARGET_SAMPLE_RATE (16kHz)
            # librosa.load can handle WAV, MP3, etc. If it fails due to ffmpeg not being installed
            # for mp3, soundfile might support wav/flac natively.
            # We enforce wav as default.
            try:
                audio, sr = sf.read(str(temp_path))
            except Exception:
                # Fallback to librosa.load if sf.read fails (e.g. for non-wav formats like MP3)
                audio, sr = librosa.load(str(temp_path), sr=None)

            if audio.ndim > 1:
                audio = audio[:, 0]

            audio = np.asarray(audio, dtype=np.float32)

            if sr != TARGET_SAMPLE_RATE:
                audio = librosa.resample(
                    audio, orig_sr=float(sr), target_sr=float(TARGET_SAMPLE_RATE)
                )

            # Normalize peak amplitude to avoid clipping and optimize signal range
            peak = float(np.max(np.abs(audio))) or 1.0
            audio_i16 = (audio * (32767.0 / peak)).astype(np.int16)

            sf.write(str(out_path), audio_i16, TARGET_SAMPLE_RATE)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        return out_path

    def synthesize_clips(
        self,
        phrases: list[str],
        output_dir: Path,
        n_samples: int,
        *,
        start_index: int = 0,
        batch_size: int = 50,
    ) -> list[Path]:
        del batch_size  # ThreadPoolExecutor concurrency handles batching/parallelism
        if not phrases:
            raise ValueError("phrases must be non-empty")
        output_dir.mkdir(parents=True, exist_ok=True)

        generated: list[Path] = []
        # Pre-populate lists for easy resumption check
        for i in range(start_index):
            p = output_dir / f"clip_{i:06d}.wav"
            if p.exists():
                generated.append(p)

        n_voices = len(self._voices)

        pbar = tqdm(
            total=n_samples,
            initial=start_index,
            desc="OpenAI TTS clips",
            unit="clip",
        )

        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            futures_to_idx = {}
            for sample_idx in range(start_index, n_samples):
                phrase = phrases[sample_idx % len(phrases)]
                voice = self._voices[sample_idx % n_voices]
                lang = (
                    self._languages[sample_idx % len(self._languages)] if self._languages else None
                )
                inst = (
                    self._instructions[sample_idx % len(self._instructions)]
                    if self._instructions
                    else None
                )

                out_path = output_dir / f"clip_{sample_idx:06d}.wav"
                fut = executor.submit(
                    self._generate_single_clip,
                    sample_idx,
                    phrase,
                    voice,
                    lang,
                    inst,
                    out_path,
                )
                futures_to_idx[fut] = sample_idx

            for fut in as_completed(futures_to_idx):
                idx = futures_to_idx[fut]
                try:
                    out_path = fut.result()
                    generated.append(out_path)
                except Exception as e:
                    logger.error("Failed to generate clip %d: %s", idx, e)
                finally:
                    pbar.update(1)

        pbar.close()
        logger.info("Generated %d clips in %s", len(generated), output_dir)
        return sorted(generated)
