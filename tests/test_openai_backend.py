"""Tests for the OpenAI-style TTS backend."""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from livekit.wakeword.config import TtsBackend, WakeWordConfig
from livekit.wakeword.data.tts import get_tts_backend
from livekit.wakeword.data.tts.openai_backend import OpenAiBackend


def generate_mock_wav_bytes(sr: int = 24000) -> bytes:
    """Generate a valid mock WAV byte stream in memory for testing."""
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)
    audio_i16 = (audio * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(audio_i16.tobytes())
    return buf.getvalue()


def test_openai_backend_config() -> None:
    """Test backend initialization and custom configuration mapping."""
    config = WakeWordConfig(
        model_name="test",
        target_phrases=["hey test"],
        tts_backend=TtsBackend.openai,
    )
    config.openai_tts.base_url = "https://my-custom-tts.api"
    config.openai_tts.voices = ["voice1", "voice2"]
    config.openai_tts.languages = ["en"]
    config.openai_tts.instructions = ["speak fast"]

    backend = get_tts_backend(config)
    assert isinstance(backend, OpenAiBackend)
    assert backend._base_url == "https://my-custom-tts.api"
    assert backend._voices == ["voice1", "voice2"]
    assert backend._languages == ["en"]
    assert backend._instructions == ["speak fast"]


@patch("urllib.request.urlopen")
def test_openai_backend_synthesis(mock_urlopen: MagicMock, tmp_path: Path) -> None:
    """Test successful clip generation and parameter cycling/payloads."""
    mock_response = MagicMock()
    mock_response.read.return_value = generate_mock_wav_bytes(sr=24000)
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    config = WakeWordConfig(
        model_name="test",
        target_phrases=["hey test"],
        tts_backend=TtsBackend.openai,
    )
    config.openai_tts.voices = ["alloy", "echo"]
    config.openai_tts.languages = ["en", "fr"]
    config.openai_tts.concurrency = 2

    backend = get_tts_backend(config)
    backend.validate_artifacts()

    output_dir = tmp_path / "clips"
    clips = backend.synthesize_clips(
        phrases=["hello", "world"],
        output_dir=output_dir,
        n_samples=4,
    )

    assert len(clips) == 4
    for clip in clips:
        assert clip.exists()
        audio, sr = sf.read(str(clip))
        assert sr == 16000
        assert audio.ndim == 1

    # Verify standard request attributes and JSON payloads
    assert mock_urlopen.call_count == 4
    calls = mock_urlopen.call_args_list
    payloads = []
    for call in calls:
        req = call[0][0]
        assert req.full_url == "https://api.openai.com/v1/audio/speech"
        payload = json.loads(req.data.decode("utf-8"))
        payloads.append(payload)

    # Check cycling parameters
    assert any(
        p["input"] == "hello" and p["voice"] == "alloy" and p["language"] == "en" for p in payloads
    )
    assert any(
        p["input"] == "world" and p["voice"] == "echo" and p["language"] == "fr" for p in payloads
    )


@patch("urllib.request.urlopen")
def test_openai_backend_retries(mock_urlopen: MagicMock, tmp_path: Path) -> None:
    """Test backend retry logic with exponential backoff on HTTP 429 errors."""
    import urllib.error

    mock_err = urllib.error.HTTPError(
        url="http://mock",
        code=429,
        msg="Too Many Requests",
        hdrs=MagicMock(),
        fp=None,
    )

    mock_response = MagicMock()
    mock_response.read.return_value = generate_mock_wav_bytes(sr=16000)
    mock_response.__enter__.return_value = mock_response

    mock_urlopen.side_effect = [mock_err, mock_response]

    config = WakeWordConfig(
        model_name="test",
        target_phrases=["hey test"],
        tts_backend=TtsBackend.openai,
    )
    config.openai_tts.max_retries = 1
    config.openai_tts.concurrency = 1

    backend = get_tts_backend(config)
    with patch("time.sleep") as mock_sleep:
        clips = backend.synthesize_clips(
            phrases=["hello"],
            output_dir=tmp_path / "clips",
            n_samples=1,
        )
        assert len(clips) == 1
        assert mock_sleep.call_count == 1
        assert mock_urlopen.call_count == 2
