from pathlib import Path

import soundfile as sf

from pipeline.audio import extract_segment, chunk_audio
from pipeline.models import Segment


class ASRTranscriber:
    """ASR using NVIDIA Canary QWEN 2.5B via NeMo SALM.

    Uses model.generate() with chat-style prompts. Max 40 seconds per chunk.
    """

    def __init__(self) -> None:
        from nemo.collections.speechlm2.models import SALM  # noqa: PLC0415

        self.model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe a single audio file (must be <= 40 seconds, 16kHz mono)."""
        answer_ids = self.model.generate(
            prompts=[
                [
                    {
                        "role": "user",
                        "content": f"Transcribe the following: {self.model.audio_locator_tag}",
                        "audio": [audio_path],
                    }
                ]
            ],
            max_new_tokens=256,
        )
        return self.model.tokenizer.ids_to_text(answer_ids[0].cpu())

    def transcribe_segments(
        self, audio_path: str, segments: list[Segment], temp_dir: str
    ) -> list[Segment]:
        """Transcribe each segment, chunking any that exceed 38 seconds."""
        for i, seg in enumerate(segments):
            seg_path = str(Path(temp_dir) / f"seg_{i:04d}.wav")
            extract_segment(audio_path, seg.start, seg.end, seg_path)

            if seg.duration > 38.0:
                chunks = chunk_audio(seg_path, max_duration=38.0)
                texts = [self.transcribe_file(cp) for cp, _, _ in chunks]
                seg.text = " ".join(texts)
            else:
                seg.text = self.transcribe_file(seg_path)

        return segments

    def unload(self) -> None:
        """Free GPU memory for sequential model loading."""
        import torch

        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
