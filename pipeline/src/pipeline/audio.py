from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.exceptions import AudioError


def split_channels(input_path: str, output_dir: str) -> tuple[str, str]:
    """Split a 2-channel WAV into two mono files.

    Returns (others_path, you_path) — channel 0 is others, channel 1 is you.
    """
    audio, sr = sf.read(input_path)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise AudioError(f"Expected 2-channel audio, got shape {audio.shape}")

    others_path = str(Path(output_dir) / "ch0_others.wav")
    you_path = str(Path(output_dir) / "ch1_you.wav")

    sf.write(others_path, audio[:, 0], sr, subtype="FLOAT")
    sf.write(you_path, audio[:, 1], sr, subtype="FLOAT")

    return others_path, you_path


def extract_segment(
    audio_path: str, start: float, end: float, output_path: str
) -> None:
    """Extract a time range from an audio file and write to output_path."""
    audio, sr = sf.read(audio_path)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = audio[start_sample:end_sample]
    sf.write(output_path, segment, sr)


def chunk_audio(
    audio_path: str, max_duration: float = 38.0
) -> list[tuple[str, float, float]]:
    """Split audio into chunks under max_duration seconds.

    Returns list of (chunk_path, start_sec, end_sec).
    Short files are returned as-is (no copy).
    """
    audio, sr = sf.read(audio_path)
    total_duration = len(audio) / sr

    if total_duration <= max_duration:
        return [(audio_path, 0.0, total_duration)]

    chunks: list[tuple[str, float, float]] = []
    chunk_samples = int(max_duration * sr)
    parent = Path(audio_path).parent

    for i, start in enumerate(range(0, len(audio), chunk_samples)):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        chunk_path = str(parent / f"chunk_{i:04d}.wav")
        sf.write(chunk_path, chunk, sr)
        chunks.append((chunk_path, start / sr, end / sr))

    return chunks
