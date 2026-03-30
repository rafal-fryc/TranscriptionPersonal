from typing import Any

import torch

from pipeline.models import Segment


class SileroVAD:
    """Lightweight VAD using Silero (~2MB model). Used for the 'You' channel."""

    def __init__(self) -> None:
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self._get_speech_timestamps: Any = self.utils[0]
        self._read_audio: Any = self.utils[2]

    def detect_speech(
        self, audio_path: str, speaker_label: str = "Speaker"
    ) -> list[Segment]:
        """Detect speech regions and return labeled segments (no text yet)."""
        wav = self._read_audio(audio_path, sampling_rate=16000)
        timestamps = self._get_speech_timestamps(
            wav, self.model, sampling_rate=16000
        )
        segments: list[Segment] = []
        for ts in timestamps:
            segments.append(
                Segment(
                    start=ts["start"] / 16000,
                    end=ts["end"] / 16000,
                    speaker=speaker_label,
                    text="",
                )
            )
        return segments
