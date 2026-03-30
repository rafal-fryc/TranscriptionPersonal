class RecorderError(Exception):
    """Base exception for recorder."""


class DeviceNotFoundError(RecorderError):
    """Audio device not found."""


class CaptureError(RecorderError):
    """Error during audio capture."""
