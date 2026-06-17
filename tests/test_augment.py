"""Tests for augmentation utilities."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from livekit.wakeword.data.augment import AudioAugmentor, align_clip_to_end


class TestAlignClipToEnd:
    def test_basic_alignment(self):
        audio = np.ones(8000, dtype=np.float32)  # 0.5s at 16kHz
        target_length = 32000  # 2s
        result = align_clip_to_end(audio, target_length, jitter_samples=0)
        assert result.shape == (target_length,)
        # Audio should be at the end
        assert np.sum(result[-8000:]) > 0
        assert np.sum(result[:16000]) == 0.0

    def test_output_length(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = align_clip_to_end(audio, 32000, jitter_samples=0)
        assert len(result) == 32000

    def test_longer_clip_than_target(self):
        audio = np.ones(48000, dtype=np.float32)  # 3s, longer than target
        result = align_clip_to_end(audio, 32000, jitter_samples=0)
        assert len(result) == 32000


class TestBackgroundMixing:
    def test_mix_level_percent_blends_original_and_background(self, tmp_path):
        bg_dir = tmp_path / "backgrounds"
        bg_dir.mkdir()
        sf.write(bg_dir / "noise.wav", np.zeros(16000, dtype=np.float32), 16000)

        augmentor = AudioAugmentor(
            background_paths=[bg_dir],
            rir_paths=[],
            background_mix_level_percent=15.0,
        )
        audio = np.ones(16000, dtype=np.float32)

        mixed = augmentor.mix_with_background(audio)

        assert np.allclose(mixed, 0.85, atol=1e-4)
