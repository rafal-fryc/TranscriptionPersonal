import argparse
import sys
from pathlib import Path

from recorder.capture import DualChannelRecorder
from recorder.config import RecorderConfig
from recorder.devices import list_devices


def main() -> None:
    parser = argparse.ArgumentParser(description="Record system audio + microphone")
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit"
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory for output WAV files"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=48000,
        help="Native device sample rate (default: 48000)",
    )
    args = parser.parse_args()

    if args.list_devices:
        devices = list_devices()
        for category, names in devices.items():
            print(f"\n{category}:")
            for name in names:
                print(f"  - {name}")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = RecorderConfig(
        output_dir=args.output_dir,
        native_sample_rate=args.sample_rate,
    )
    recorder = DualChannelRecorder(config)

    print("Recording... Press Enter to stop.")
    recorder.start()

    try:
        input()
    except KeyboardInterrupt:
        print()

    output = recorder.stop()
    size_mb = output.stat().st_size / 1024 / 1024
    print(f"Saved: {output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
