import tempfile
from pathlib import Path

from pipeline.audio import split_channels
from pipeline.config import PipelineConfig
from pipeline.diarizer import SpeakerDiarizer
from pipeline.formatter import format_transcript
from pipeline.models import Segment
from pipeline.transcriber import ASRTranscriber
from pipeline.vad import SileroVAD


def process_recording(
    recording_path: str,
    output_path: str,
    config: PipelineConfig,
) -> str:
    """Full pipeline: split → segment → transcribe → format.

    Models are loaded and unloaded sequentially to fit in 8GB.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Split 2-channel WAV into mono files
        others_path, you_path = split_channels(recording_path, temp_dir)

        # Step 2: Detect speech in "You" channel (Silero VAD — tiny, always fits)
        vad = SileroVAD()
        you_segments = [
            Segment(start=s.start, end=s.end, speaker="You", text=s.text)
            for s in vad.detect_speech(you_path, speaker_label="You")
        ]

        # Step 3: Segment "Others" channel
        if config.enable_diarization and config.hf_token:
            diarizer = SpeakerDiarizer(config.hf_token)
            others_segments = diarizer.diarize(others_path)
            diarizer.unload()
        else:
            others_segments = [
                Segment(start=s.start, end=s.end, speaker="Other", text=s.text)
                for s in vad.detect_speech(others_path, speaker_label="Other")
            ]

        del vad

        # Step 4: Transcribe all segments (load ASR model after diarizer is unloaded)
        transcriber = ASRTranscriber()
        you_segments = transcriber.transcribe_segments(
            you_path, you_segments, temp_dir
        )
        others_segments = transcriber.transcribe_segments(
            others_path, others_segments, temp_dir
        )
        transcriber.unload()

        # Step 5: Assemble and write transcript
        all_segments = you_segments + others_segments
        transcript = format_transcript(all_segments)

        Path(output_path).write_text(transcript, encoding="utf-8")
        return transcript
