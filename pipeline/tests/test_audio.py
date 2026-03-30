import numpy as np
import numpy.testing as npt
import soundfile as sf
from pathlib import Path
import pytest

from pipeline.audio import split_channels, extract_segment, chunk_audio
from pipeline.exceptions import AudioError


def _make_stereo_wav(path: str, sr: int = 16000, duration: float = 1.0) -> None:
    """Helper: create a 2-channel WAV with distinct tones per channel."""
    samples = int(sr * duration)
    t = np.arange(samples) / sr
    ch0 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    ch1 = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    sf.write(path, np.column_stack([ch0, ch1]), sr, subtype="FLOAT")


class TestSplitChannels:
    def test_should_produce_two_mono_files(self, temp_dir: str) -> None:
        input_path = str(Path(temp_dir) / "stereo.wav")
        _make_stereo_wav(input_path)
        others_path, you_path = split_channels(input_path, temp_dir)
        others, sr0 = sf.read(others_path)
        you, sr1 = sf.read(you_path)
        assert others.ndim == 1
        assert you.ndim == 1
        assert sr0 == 16000
        assert sr1 == 16000

    def test_should_preserve_channel_content(self, temp_dir: str) -> None:
        input_path = str(Path(temp_dir) / "stereo.wav")
        sr = 16000
        samples = sr
        ch0 = np.ones(samples, dtype=np.float32)
        ch1 = np.full(samples, 0.5, dtype=np.float32)
        sf.write(input_path, np.column_stack([ch0, ch1]), sr, subtype="FLOAT")

        others_path, you_path = split_channels(input_path, temp_dir)
        others, _ = sf.read(others_path)
        you, _ = sf.read(you_path)
        npt.assert_allclose(others, ch0, atol=1e-6)
        npt.assert_allclose(you, ch1, atol=1e-6)

    def test_should_raise_for_mono_input(self, temp_dir: str) -> None:
        input_path = str(Path(temp_dir) / "mono.wav")
        sf.write(input_path, np.zeros(16000, dtype=np.float32), 16000)
        with pytest.raises(AudioError):
            split_channels(input_path, temp_dir)


class TestExtractSegment:
    def test_should_extract_time_range(self, temp_dir: str) -> None:
        input_path = str(Path(temp_dir) / "full.wav")
        sr = 16000
        audio = np.arange(sr * 4, dtype=np.float32) / (sr * 4)
        sf.write(input_path, audio, sr)

        output_path = str(Path(temp_dir) / "seg.wav")
        extract_segment(input_path, 1.0, 3.0, output_path)
        seg, seg_sr = sf.read(output_path)

        assert seg_sr == sr
        assert len(seg) == sr * 2


class TestChunkAudio:
    def test_short_audio_returns_single_chunk(self, temp_dir: str) -> None:
        path = str(Path(temp_dir) / "short.wav")
        sf.write(path, np.zeros(16000 * 10, dtype=np.float32), 16000)
        chunks = chunk_audio(path, max_duration=38.0)
        assert len(chunks) == 1
        assert chunks[0][0] == path

    def test_long_audio_returns_multiple_chunks(self, temp_dir: str) -> None:
        path = str(Path(temp_dir) / "long.wav")
        sf.write(path, np.zeros(16000 * 80, dtype=np.float32), 16000)
        chunks = chunk_audio(path, max_duration=38.0)
        assert len(chunks) == 3
        for chunk_path, start, end in chunks:
            assert Path(chunk_path).exists()
            assert end > start
