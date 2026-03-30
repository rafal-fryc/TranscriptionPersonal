class PipelineError(Exception):
    """Base exception for pipeline."""


class AudioError(PipelineError):
    """Error processing audio."""
