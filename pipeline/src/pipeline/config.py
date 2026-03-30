from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    hf_token: str = ""
    enable_diarization: bool = True
    max_segment_duration: float = 38.0
