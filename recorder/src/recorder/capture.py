import threading
from datetime import datetime
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from recorder.config import RecorderConfig
from recorder.devices import get_loopback_device, get_microphone
from recorder.exceptions import CaptureError


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from orig_sr to target_sr using polyphase filtering."""
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


class DualChannelRecorder:
    def __init__(self, config: RecorderConfig | None = None) -> None:
        self.config = config or RecorderConfig()
        self._recording = False
        self._loopback_chunks: list[np.ndarray] = []
        self._mic_chunks: list[np.ndarray] = []
        self._loopback_thread: threading.Thread | None = None
        self._mic_thread: threading.Thread | None = None

    def _record_stream(
        self,
        device: Any,
        chunks: list[np.ndarray],
    ) -> None:
        native_sr = self.config.native_sample_rate
        chunk_frames = int(native_sr * 0.5)
        try:
            # Must record stereo on Windows WASAPI — mono gives corrupted data
            with device.recorder(samplerate=native_sr, channels=2) as rec:
                while self._recording:
                    data = rec.record(numframes=chunk_frames)
                    mono = data.mean(axis=1).astype(np.float32)
                    chunks.append(mono)
        except Exception as e:
            if self._recording:
                raise CaptureError(f"Stream recording failed: {e}") from e

    def start(self) -> None:
        """Start recording from loopback and microphone."""
        self._recording = True
        self._loopback_chunks = []
        self._mic_chunks = []

        loopback = get_loopback_device()
        mic = get_microphone()

        self._loopback_thread = threading.Thread(
            target=self._record_stream,
            args=(loopback, self._loopback_chunks),
            daemon=True,
        )
        self._mic_thread = threading.Thread(
            target=self._record_stream,
            args=(mic, self._mic_chunks),
            daemon=True,
        )
        self._loopback_thread.start()
        self._mic_thread.start()

    def stop(self) -> Path:
        """Stop recording and save to WAV. Returns the output file path."""
        self._recording = False
        if self._loopback_thread:
            self._loopback_thread.join(timeout=5)
        if self._mic_thread:
            self._mic_thread.join(timeout=5)
        return self._save()

    def _save(self) -> Path:
        """Resample, align, and write 2-channel WAV."""
        if not self._loopback_chunks or not self._mic_chunks:
            raise CaptureError("No audio data recorded")

        native_sr = self.config.native_sample_rate
        target_sr = self.config.target_sample_rate

        loopback_audio = _resample(np.concatenate(self._loopback_chunks), native_sr, target_sr)
        mic_audio = _resample(np.concatenate(self._mic_chunks), native_sr, target_sr)

        min_len = min(len(loopback_audio), len(mic_audio))
        stereo = np.column_stack([
            loopback_audio[:min_len],
            mic_audio[:min_len],
        ])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config.output_dir) / f"{self.config.file_prefix}_{timestamp}.wav"
        sf.write(str(output_path), stereo, target_sr)

        return output_path
