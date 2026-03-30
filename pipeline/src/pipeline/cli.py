import argparse
import os
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.orchestrator import process_recording


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a 2-channel recording")
    parser.add_argument("input", help="Path to 2-channel WAV file")
    parser.add_argument(
        "-o", "--output", default=None, help="Output transcript path (default: <input>_transcript.txt)"
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace token for pyannote (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip speaker diarization, label all others as 'Other'",
    )
    args = parser.parse_args()

    output = args.output or str(Path(args.input).stem) + "_transcript.txt"
    config = PipelineConfig(
        hf_token=args.hf_token,
        enable_diarization=not args.no_diarize,
    )

    print(f"Processing: {args.input}")
    if config.enable_diarization:
        print("Mode: full diarization (sequential model loading)")
    else:
        print("Mode: VAD only (You / Other labels)")

    transcript = process_recording(args.input, output, config)

    print(f"\nTranscript saved to: {output}")
    print(f"\n--- Preview (first 500 chars) ---\n{transcript[:500]}")


if __name__ == "__main__":
    main()
