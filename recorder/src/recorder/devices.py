from typing import Any

import soundcard as sc

from recorder.exceptions import DeviceNotFoundError

# soundcard exposes platform-specific internal types (e.g. mediafoundation._Microphone)
# that are not part of the public API. Using Any here to interface with the untyped
# third-party library.
_Microphone = Any


def get_loopback_device() -> _Microphone:
    """Get the WASAPI loopback device for the default speaker."""
    speaker = sc.default_speaker()
    try:
        return sc.get_microphone(id=str(speaker.name), include_loopback=True)
    except Exception as e:
        raise DeviceNotFoundError(
            f"No loopback device found for speaker: {speaker.name}"
        ) from e


def get_microphone() -> _Microphone:
    """Get the default microphone."""
    mic = sc.default_microphone()
    if mic is None:
        raise DeviceNotFoundError("No default microphone found")
    return mic


def list_devices() -> dict[str, list[str]]:
    """List all audio devices by category."""
    speakers = [s.name for s in sc.all_speakers()]
    mics = [m.name for m in sc.all_microphones(include_loopback=False)]
    all_mics = sc.all_microphones(include_loopback=True)
    loopbacks = [m.name for m in all_mics if getattr(m, "isloopback", False)]
    return {
        "speakers": speakers,
        "microphones": mics,
        "loopback_devices": loopbacks,
    }
