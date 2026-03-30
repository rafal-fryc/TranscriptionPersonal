from unittest.mock import MagicMock, patch
import numpy as np
import numpy.testing as npt
import soundfile as sf
from pathlib import Path
import pytest

from recorder.capture import DualChannelRecorder, _resample
from recorder.config import RecorderConfig
from recorder.exceptions import CaptureError


class TestResample:
    def test_should_downsample_48k_to_16k(self) -> None:
        orig_sr = 48000
        target_sr = 16000
        duration = 1.0
        t = np.arange(int(orig_sr * duration)) / orig_sr
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        result = _resample(audio, orig_sr, target_sr)
        expected_len = int(target_sr * duration)
        assert len(result) == expected_len

    def test_should_return_unchanged_when_same_rate(self) -> None:
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _resample(audio, 16000, 16000)
        npt.assert_array_equal(result, audio)


class TestDualChannelRecorder:
    def test_should_produce_two_channel_16khz_wav(self, temp_dir: str) -> None:
        config = RecorderConfig(
            output_dir=temp_dir,
            target_sample_rate=16000,
            native_sample_rate=48000,
        )
        recorder = DualChannelRecorder(config)

        sr = 48000
        chunk_samples = int(sr * 0.5)
        chunk_loopback = np.random.randn(chunk_samples).astype(np.float32)
        chunk_mic = np.random.randn(chunk_samples).astype(np.float32)

        recorder._loopback_chunks = [chunk_loopback, chunk_loopback]
        recorder._mic_chunks = [chunk_mic, chunk_mic]

        output_path = recorder._save()

        audio, file_sr = sf.read(str(output_path))
        assert file_sr == 16000
        assert audio.ndim == 2
        assert audio.shape[1] == 2

    def test_should_align_channels_to_shorter_length(self, temp_dir: str) -> None:
        config = RecorderConfig(
            output_dir=temp_dir,
            target_sample_rate=16000,
            native_sample_rate=16000,
        )
        recorder = DualChannelRecorder(config)

        recorder._loopback_chunks = [np.ones(1000, dtype=np.float32)]
        recorder._mic_chunks = [np.ones(800, dtype=np.float32)]

        output_path = recorder._save()

        audio, _ = sf.read(str(output_path))
        assert audio.shape[0] == 800
        assert audio.shape[1] == 2

    def test_should_raise_when_no_data(self, temp_dir: str) -> None:
        config = RecorderConfig(output_dir=temp_dir)
        recorder = DualChannelRecorder(config)
        recorder._loopback_chunks = []
        recorder._mic_chunks = []

        with pytest.raises(CaptureError):
            recorder._save()
