from dataclasses import dataclass


@dataclass(frozen=True)
class RecorderConfig:
    output_dir: str = "."
    target_sample_rate: int = 16000
    file_prefix: str = "recording"
    native_sample_rate: int = 48000
