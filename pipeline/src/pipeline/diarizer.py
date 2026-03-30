import torch
from pyannote.audio import Pipeline as PyannotePipeline

from pipeline.models import Segment


class SpeakerDiarizer:
    """Speaker diarization using pyannote 3.1. Used for the 'Others' channel.

    Requires a HuggingFace token with access to:
    - pyannote/speaker-diarization-3.1
    - pyannote/segmentation-3.0
    Accept user agreements on both model pages first.
    """

    def __init__(self, hf_token: str) -> None:
        self.pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

    def diarize(self, audio_path: str) -> list[Segment]:
        """Run diarization and return speaker-labeled segments (no text yet)."""
        result = self.pipeline(audio_path)
        segments: list[Segment] = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            segments.append(
                Segment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                    text="",
                )
            )
        return segments

    def unload(self) -> None:
        """Free GPU memory for sequential model loading."""
        del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
