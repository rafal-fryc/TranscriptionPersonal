# Local Transcription System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-component system: a Windows laptop recorder that captures system audio + microphone into a 2-channel WAV, and a Jetson Orin Nano pipeline that diarizes and transcribes the recording into a speaker-labeled text transcript.

**Architecture:** The recorder captures WASAPI loopback (others' audio) and microphone (your audio) as two channels in a single WAV file. After the call, the file is transferred to the Jetson, where models are loaded sequentially (one at a time) to fit in 8GB shared memory. Silero VAD detects your speech segments, pyannote diarizes the others' channel, and Canary QWEN 2.5B transcribes everything.

**Tech Stack:**
- Recorder: Python, `soundcard` (WASAPI loopback), `scipy` (resampling), `soundfile` (WAV I/O)
- Pipeline: Python, NeMo `SALM` (Canary ASR), `pyannote.audio` (diarization), Silero VAD (`torch.hub`), `soundfile`
- Transfer: `scp` over SSH/Tailscale

---

## Critical Constraint: 8GB Shared Memory

The Jetson Orin Nano has 8GB shared between CPU and GPU. pyannote diarization 3.1 can use 4-6GB+ for processing, and Canary QWEN 2.5B needs ~3-4GB in INT8. They cannot coexist in memory.

**Solution: sequential model loading.** Load pyannote, diarize, unload and free GPU memory, then load Canary, transcribe, unload. Peak usage stays at ~4-5GB.

---

## File Structure

```
transcription/
├── recorder/                          # Deployed to Windows laptop
│   ├── pyproject.toml
│   ├── src/
│   │   └── recorder/
│   │       ├── __init__.py
│   │       ├── config.py             # RecorderConfig dataclass
│   │       ├── exceptions.py         # RecorderError, DeviceNotFoundError, CaptureError
│   │       ├── devices.py            # Device enumeration, loopback/mic discovery
│   │       ├── capture.py            # DualChannelRecorder (threading, resample, WAV output)
│   │       └── cli.py                # Entry point: start/stop recording
│   └── tests/
│       ├── conftest.py               # Shared fixtures (temp dirs, mock devices)
│       ├── test_devices.py           # Device discovery tests (mocked soundcard)
│       └── test_capture.py           # Capture logic tests (resampling, channel alignment)
├── pipeline/                          # Deployed to Jetson Orin Nano
│   ├── pyproject.toml
│   ├── src/
│   │   └── pipeline/
│   │       ├── __init__.py
│   │       ├── config.py             # PipelineConfig dataclass
│   │       ├── exceptions.py         # PipelineError, AudioError
│   │       ├── models.py             # Segment dataclass
│   │       ├── formatter.py          # Transcript text assembly + formatting
│   │       ├── audio.py              # Channel splitting, segment extraction, chunking
│   │       ├── vad.py                # Silero VAD wrapper
│   │       ├── diarizer.py           # pyannote diarization wrapper
│   │       ├── transcriber.py        # NeMo Canary SALM wrapper
│   │       ├── orchestrator.py       # Full pipeline: split → diarize → transcribe → format
│   │       └── cli.py                # Entry point: process WAV → transcript
│   └── tests/
│       ├── conftest.py               # Shared fixtures (test WAVs, temp dirs)
│       ├── test_models.py            # Segment dataclass tests
│       ├── test_formatter.py         # Transcript formatting tests
│       ├── test_audio.py             # Channel split + chunking tests
│       └── test_orchestrator.py      # Pipeline flow tests (mocked models)
├── scripts/
│   └── transfer.sh                   # SCP file to Jetson
├── DECISIONS.md
├── .gitignore
└── transcription-notes.md            # (exists)
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `recorder/pyproject.toml`
- Create: `recorder/src/recorder/__init__.py`
- Create: `pipeline/pyproject.toml`
- Create: `pipeline/src/pipeline/__init__.py`
- Create: `.gitignore`
- Create: `DECISIONS.md`

- [ ] **Step 1: Initialize git repository**

```bash
cd C:\Users\rafst\documents\projectas\transcription
git init
```

Expected: `Initialized empty Git repository`

- [ ] **Step 2: Create .gitignore**

Create `.gitignore`:

```gitignore
.env
.env.local
.env.*.local
*.log
.DS_Store
Thumbs.db

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/

# Audio files (large, don't commit)
*.wav
*.flac
*.mp3

# IDE
.vscode/
.idea/

# Temp
*.tmp
```

- [ ] **Step 3: Create recorder/pyproject.toml**

```toml
[project]
name = "transcription-recorder"
version = "0.1.0"
description = "Two-channel audio recorder for local transcription"
requires-python = ">=3.11"
dependencies = [
    "soundcard>=0.4.0",
    "soundfile>=0.12.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
]

[project.scripts]
record = "recorder.cli:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.mypy]
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create pipeline/pyproject.toml**

```toml
[project]
name = "transcription-pipeline"
version = "0.1.0"
description = "Speech transcription pipeline for Jetson Orin Nano"
requires-python = ">=3.11"
dependencies = [
    "soundfile>=0.12.0",
    "numpy>=1.24.0",
    "torch>=2.0.0",
]

[project.optional-dependencies]
diarization = [
    "pyannote.audio>=3.1.0",
]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
]

[project.scripts]
transcribe = "pipeline.cli:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.mypy]
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Note: NeMo is installed separately via `pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"`. Silero VAD is loaded via `torch.hub` (no pip install). On Jetson, use JetPack SDK's prebuilt PyTorch wheels.

- [ ] **Step 5: Create package __init__.py files and directory structure**

```bash
mkdir -p recorder/src/recorder recorder/tests
mkdir -p pipeline/src/pipeline pipeline/tests
touch recorder/src/recorder/__init__.py
touch pipeline/src/pipeline/__init__.py
```

- [ ] **Step 6: Create DECISIONS.md**

Create `DECISIONS.md`:

```markdown
# Decisions

## 2026-03-28 — Canary QWEN 2.5B over Cohere Transcribe
**Chosen:** nvidia/canary-qwen-2.5b for ASR
**Alternatives:** CohereLabs/cohere-transcribe-03-2026 (2B params, 14 languages, 5.42% avg WER)
**Why:** NVIDIA-on-NVIDIA toolchain. Canary is a NeMo model with direct nemo2onnx → TensorRT export path, which is NVIDIA's own inference stack built for Jetson. Cohere requires HuggingFace Transformers (>=5.4.0) with weaker Jetson support and no native TensorRT path. NVIDIA's INT8 quantization tooling is more mature for their own models.
**Trade-offs:** English only (Cohere supports 14 languages). Slightly worse raw WER benchmarks pre-quantization.
**Revisit if:** Multilingual support needed, or Cohere adds TensorRT export.

## 2026-03-28 — Sequential model loading on Jetson
**Chosen:** Load one model at a time (diarize → unload → ASR → unload)
**Alternatives:** Both models resident simultaneously; offload diarization to CPU
**Why:** Jetson Orin Nano has 8GB shared CPU/GPU memory. pyannote 3.1 uses 4-6GB+, Canary needs 3-4GB INT8. Cannot coexist. Sequential loading keeps peak at ~4-5GB.
**Trade-offs:** Slower processing (model load/unload overhead). Cannot pipeline diarization and ASR in parallel.
**Revisit if:** Upgrading to a device with more memory, or if lighter diarization models become available.

## 2026-03-28 — Silero VAD for "You" channel, pyannote for "Others" channel
**Chosen:** Silero VAD (~2MB) for your mic channel; pyannote speaker-diarization-3.1 for others' channel
**Alternatives:** pyannote on both channels; energy-based VAD
**Why:** Your mic is a single speaker — full diarization is wasted compute. Silero is tiny and accurate. Others' channel needs real speaker separation when multiple participants are on the call.
**Trade-offs:** If the "You" channel picks up cross-talk from speakers, Silero won't distinguish speakers. Acceptable since the mic primarily captures your voice.
**Revisit if:** Cross-talk is a frequent problem in practice.

## 2026-03-28 — soundcard library for WASAPI loopback capture
**Chosen:** Python `soundcard` library
**Alternatives:** pyaudiowpatch (PyAudio fork), sounddevice (PortAudio), ffmpeg CLI
**Why:** Cleanest Python API for loopback recording (`get_microphone(include_loopback=True)`). Well-maintained, cross-platform (though we only need Windows).
**Trade-offs:** Must record stereo on Windows (mono loopback gives corrupted data). No auto-resampling via WASAPI — must resample manually. Silence at recording start can cause issues.
**Revisit if:** soundcard has persistent issues on user's Windows audio setup; pyaudiowpatch as fallback.

## 2026-03-28 — Batch post-call transcription over real-time streaming
**Chosen:** Record locally during call, transfer and transcribe after call ends
**Alternatives:** Stream audio to Jetson in real-time for live transcription
**Why:** No network dependency during calls (can't drop or lag). Simpler code (~50 lines recorder vs streaming infrastructure). Batch processing is faster (can chunk in parallel). Raw audio preserved for re-transcription if models improve.
**Trade-offs:** No live transcript during the call. Must wait for processing after call ends.
**Revisit if:** User needs live captions during calls.

## 2026-03-28 — Two-package monorepo
**Chosen:** Single git repo with `recorder/` and `pipeline/` packages
**Alternatives:** Separate repos for each component
**Why:** Both components share the audio format contract (2-ch, 16kHz, 16-bit WAV) and are part of the same workflow. Single repo keeps the contract co-located and simplifies versioning.
**Trade-offs:** Deployment to different machines requires copying the relevant subdirectory.
**Revisit if:** Components diverge significantly in development cadence or team ownership.
```

- [ ] **Step 7: Commit scaffolding**

```bash
git add .gitignore DECISIONS.md recorder/pyproject.toml recorder/src/recorder/__init__.py pipeline/pyproject.toml pipeline/src/pipeline/__init__.py
git commit -m "chore: scaffold project structure with recorder and pipeline packages"
```

---

## Task 2: Recorder — Exceptions and Config

**Files:**
- Create: `recorder/src/recorder/exceptions.py`
- Create: `recorder/src/recorder/config.py`

- [ ] **Step 1: Create recorder exceptions**

Create `recorder/src/recorder/exceptions.py`:

```python
class RecorderError(Exception):
    """Base exception for recorder."""


class DeviceNotFoundError(RecorderError):
    """Audio device not found."""


class CaptureError(RecorderError):
    """Error during audio capture."""
```

- [ ] **Step 2: Create recorder config**

Create `recorder/src/recorder/config.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RecorderConfig:
    output_dir: str = "."
    target_sample_rate: int = 16000
    file_prefix: str = "recording"
    native_sample_rate: int = 48000
```

`native_sample_rate` defaults to 48000 (most common on modern Windows). The recorder captures at this rate and resamples to `target_sample_rate` when saving.

- [ ] **Step 3: Commit**

```bash
git add recorder/src/recorder/exceptions.py recorder/src/recorder/config.py
git commit -m "feat(recorder): add config and exception classes"
```

---

## Task 3: Recorder — Device Discovery

**Files:**
- Create: `recorder/src/recorder/devices.py`
- Test: `recorder/tests/test_devices.py`

- [ ] **Step 1: Write the failing tests**

Create `recorder/tests/test_devices.py`:

```python
from unittest.mock import MagicMock, patch
import pytest

from recorder.devices import get_loopback_device, get_microphone, list_devices
from recorder.exceptions import DeviceNotFoundError


@pytest.fixture
def mock_speaker() -> MagicMock:
    speaker = MagicMock()
    speaker.name = "Speakers (Realtek Audio)"
    return speaker


@pytest.fixture
def mock_loopback() -> MagicMock:
    loopback = MagicMock()
    loopback.name = "Speakers (Realtek Audio)"
    loopback.isloopback = True
    return loopback


@pytest.fixture
def mock_mic() -> MagicMock:
    mic = MagicMock()
    mic.name = "Microphone (Realtek Audio)"
    mic.isloopback = False
    return mic


class TestGetLoopbackDevice:
    @patch("recorder.devices.sc")
    def test_should_return_loopback_for_default_speaker(
        self, mock_sc: MagicMock, mock_speaker: MagicMock, mock_loopback: MagicMock
    ) -> None:
        mock_sc.default_speaker.return_value = mock_speaker
        mock_sc.get_microphone.return_value = mock_loopback
        result = get_loopback_device()
        assert result == mock_loopback
        mock_sc.get_microphone.assert_called_once_with(
            id=str(mock_speaker.name), include_loopback=True
        )

    @patch("recorder.devices.sc")
    def test_should_raise_when_no_loopback_found(
        self, mock_sc: MagicMock, mock_speaker: MagicMock
    ) -> None:
        mock_sc.default_speaker.return_value = mock_speaker
        mock_sc.get_microphone.side_effect = RuntimeError("not found")
        with pytest.raises(DeviceNotFoundError):
            get_loopback_device()


class TestGetMicrophone:
    @patch("recorder.devices.sc")
    def test_should_return_default_mic(
        self, mock_sc: MagicMock, mock_mic: MagicMock
    ) -> None:
        mock_sc.default_microphone.return_value = mock_mic
        result = get_microphone()
        assert result == mock_mic

    @patch("recorder.devices.sc")
    def test_should_raise_when_no_mic(self, mock_sc: MagicMock) -> None:
        mock_sc.default_microphone.return_value = None
        with pytest.raises(DeviceNotFoundError):
            get_microphone()


class TestListDevices:
    @patch("recorder.devices.sc")
    def test_should_list_all_device_categories(
        self,
        mock_sc: MagicMock,
        mock_speaker: MagicMock,
        mock_mic: MagicMock,
        mock_loopback: MagicMock,
    ) -> None:
        mock_sc.all_speakers.return_value = [mock_speaker]
        mock_sc.all_microphones.side_effect = [
            [mock_mic],          # include_loopback=False
            [mock_mic, mock_loopback],  # include_loopback=True
        ]
        result = list_devices()
        assert "speakers" in result
        assert "microphones" in result
        assert "loopback_devices" in result
        assert len(result["speakers"]) == 1
        assert len(result["microphones"]) == 1
        assert len(result["loopback_devices"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd C:\Users\rafst\documents\projectas\transcription\recorder
pip install -e ".[dev]" && pytest tests/test_devices.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'recorder.devices'`

- [ ] **Step 3: Write the implementation**

Create `recorder/src/recorder/devices.py`:

```python
import soundcard as sc

from recorder.exceptions import DeviceNotFoundError


def get_loopback_device() -> sc.Microphone:
    """Get the WASAPI loopback device for the default speaker."""
    speaker = sc.default_speaker()
    try:
        return sc.get_microphone(id=str(speaker.name), include_loopback=True)
    except Exception as e:
        raise DeviceNotFoundError(
            f"No loopback device found for speaker: {speaker.name}"
        ) from e


def get_microphone() -> sc.Microphone:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd C:\Users\rafst\documents\projectas\transcription\recorder
pytest tests/test_devices.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add recorder/src/recorder/devices.py recorder/tests/test_devices.py
git commit -m "feat(recorder): add audio device discovery with loopback support"
```

---

## Task 4: Recorder — Dual-Channel Capture Engine

**Files:**
- Create: `recorder/src/recorder/capture.py`
- Create: `recorder/tests/conftest.py`
- Test: `recorder/tests/test_capture.py`

- [ ] **Step 1: Create test fixtures**

Create `recorder/tests/conftest.py`:

```python
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield d
```

- [ ] **Step 2: Write the failing tests**

Create `recorder/tests/test_capture.py`:

```python
from unittest.mock import MagicMock, patch
import numpy as np
import numpy.testing as npt
import soundfile as sf
from pathlib import Path

from recorder.capture import DualChannelRecorder, _resample
from recorder.config import RecorderConfig
from recorder.exceptions import CaptureError


class TestResample:
    def test_should_downsample_48k_to_16k(self) -> None:
        orig_sr = 48000
        target_sr = 16000
        duration = 1.0
        t = np.arange(int(orig_sr * duration)) / orig_sr
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        result = _resample(audio, orig_sr, target_sr)
        expected_len = int(target_sr * duration)
        assert len(result) == expected_len

    def test_should_return_unchanged_when_same_rate(self) -> None:
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _resample(audio, 16000, 16000)
        npt.assert_array_equal(result, audio)


class TestDualChannelRecorder:
    def test_should_produce_two_channel_16khz_wav(self, temp_dir: str) -> None:
        config = RecorderConfig(
            output_dir=temp_dir,
            target_sample_rate=16000,
            native_sample_rate=48000,
        )
        recorder = DualChannelRecorder(config)

        # Simulate recording by injecting pre-recorded chunks
        sr = 48000
        chunk_samples = int(sr * 0.5)
        chunk_loopback = np.random.randn(chunk_samples).astype(np.float32)
        chunk_mic = np.random.randn(chunk_samples).astype(np.float32)

        recorder._loopback_chunks = [chunk_loopback, chunk_loopback]
        recorder._mic_chunks = [chunk_mic, chunk_mic]

        output_path = recorder._save()

        audio, file_sr = sf.read(str(output_path))
        assert file_sr == 16000
        assert audio.ndim == 2
        assert audio.shape[1] == 2

    def test_should_align_channels_to_shorter_length(self, temp_dir: str) -> None:
        config = RecorderConfig(
            output_dir=temp_dir,
            target_sample_rate=16000,
            native_sample_rate=16000,
        )
        recorder = DualChannelRecorder(config)

        recorder._loopback_chunks = [np.ones(1000, dtype=np.float32)]
        recorder._mic_chunks = [np.ones(800, dtype=np.float32)]

        output_path = recorder._save()

        audio, _ = sf.read(str(output_path))
        assert audio.shape[0] == 800
        assert audio.shape[1] == 2

    def test_should_raise_when_no_data(self, temp_dir: str) -> None:
        config = RecorderConfig(output_dir=temp_dir)
        recorder = DualChannelRecorder(config)
        recorder._loopback_chunks = []
        recorder._mic_chunks = []

        import pytest
        with pytest.raises(CaptureError):
            recorder._save()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd C:\Users\rafst\documents\projectas\transcription\recorder
pytest tests/test_capture.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'recorder.capture'`

- [ ] **Step 4: Write the implementation**

Create `recorder/src/recorder/capture.py`:

```python
import threading
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np
import soundcard as sc
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
        device: sc.Microphone,
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd C:\Users\rafst\documents\projectas\transcription\recorder
pytest tests/test_capture.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add recorder/src/recorder/capture.py recorder/tests/conftest.py recorder/tests/test_capture.py
git commit -m "feat(recorder): add dual-channel capture with WASAPI loopback and resampling"
```

---

## Task 5: Recorder — CLI Interface

**Files:**
- Create: `recorder/src/recorder/cli.py`

- [ ] **Step 1: Write the CLI**

Create `recorder/src/recorder/cli.py`:

```python
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
```

- [ ] **Step 2: Manual integration test on laptop**

```bash
cd C:\Users\rafst\documents\projectas\transcription\recorder
pip install -e .

# List devices first
record --list-devices

# Record a short test (play some audio on laptop, speak into mic, press Enter)
record --output-dir ./test-recordings

# Verify the output file
python -c "import soundfile as sf; d, sr = sf.read('test-recordings/<filename>.wav'); print(f'Rate: {sr}, Shape: {d.shape}, Duration: {len(d)/sr:.1f}s')"
```

Expected: 2-channel WAV at 16000 Hz, with audible audio on both channels.

- [ ] **Step 3: Commit**

```bash
git add recorder/src/recorder/cli.py
git commit -m "feat(recorder): add CLI with device listing and recording commands"
```

---

## Task 6: Pipeline — Exceptions, Config, and Data Models

**Files:**
- Create: `pipeline/src/pipeline/exceptions.py`
- Create: `pipeline/src/pipeline/config.py`
- Create: `pipeline/src/pipeline/models.py`
- Test: `pipeline/tests/test_models.py`

- [ ] **Step 1: Write the failing test for Segment**

Create `pipeline/tests/test_models.py`:

```python
from pipeline.models import Segment


class TestSegment:
    def test_should_store_fields(self) -> None:
        seg = Segment(start=1.5, end=4.0, speaker="You", text="hello world")
        assert seg.start == 1.5
        assert seg.end == 4.0
        assert seg.speaker == "You"
        assert seg.text == "hello world"

    def test_should_calculate_duration(self) -> None:
        seg = Segment(start=10.0, end=25.5, speaker="A", text="")
        assert seg.duration == 15.5

    def test_duration_should_be_zero_for_empty_segment(self) -> None:
        seg = Segment(start=5.0, end=5.0, speaker="A", text="")
        assert seg.duration == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:\Users\rafst\documents\projectas\transcription\pipeline
pip install -e ".[dev]" && pytest tests/test_models.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write exceptions, config, and models**

Create `pipeline/src/pipeline/exceptions.py`:

```python
class PipelineError(Exception):
    """Base exception for pipeline."""


class AudioError(PipelineError):
    """Error processing audio."""
```

Create `pipeline/src/pipeline/config.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    hf_token: str = ""
    enable_diarization: bool = True
    max_segment_duration: float = 38.0
```

Create `pipeline/src/pipeline/models.py`:

```python
from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd C:\Users\rafst\documents\projectas\transcription\pipeline
pytest tests/test_models.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/src/pipeline/exceptions.py pipeline/src/pipeline/config.py pipeline/src/pipeline/models.py pipeline/tests/test_models.py
git commit -m "feat(pipeline): add data models, config, and exceptions"
```

---

## Task 7: Pipeline — Transcript Formatter

**Files:**
- Create: `pipeline/src/pipeline/formatter.py`
- Test: `pipeline/tests/test_formatter.py`

- [ ] **Step 1: Write the failing tests**

Create `pipeline/tests/test_formatter.py`:

```python
from pipeline.formatter import format_transcript, _format_time
from pipeline.models import Segment


class TestFormatTime:
    def test_zero(self) -> None:
        assert _format_time(0.0) == "00:00:00"

    def test_seconds_only(self) -> None:
        assert _format_time(45.0) == "00:00:45"

    def test_minutes_and_seconds(self) -> None:
        assert _format_time(125.0) == "00:02:05"

    def test_hours(self) -> None:
        assert _format_time(3661.0) == "01:01:01"

    def test_fractional_seconds_truncate(self) -> None:
        assert _format_time(5.9) == "00:00:05"


class TestFormatTranscript:
    def test_single_segment(self) -> None:
        segments = [Segment(start=5.0, end=8.0, speaker="You", text="hello world")]
        result = format_transcript(segments)
        assert result == "[00:00:05] You: hello world"

    def test_should_sort_by_start_time(self) -> None:
        segments = [
            Segment(start=10.0, end=12.0, speaker="Other", text="second"),
            Segment(start=2.0, end=5.0, speaker="You", text="first"),
        ]
        result = format_transcript(segments)
        lines = result.split("\n")
        assert lines[0] == "[00:00:02] You: first"
        assert lines[1] == "[00:00:10] Other: second"

    def test_should_skip_empty_text(self) -> None:
        segments = [
            Segment(start=0.0, end=1.0, speaker="You", text=""),
            Segment(start=2.0, end=3.0, speaker="Other", text="hello"),
        ]
        result = format_transcript(segments)
        assert result == "[00:00:02] Other: hello"

    def test_should_skip_whitespace_only_text(self) -> None:
        segments = [
            Segment(start=0.0, end=1.0, speaker="You", text="   "),
            Segment(start=2.0, end=3.0, speaker="Other", text="hello"),
        ]
        result = format_transcript(segments)
        assert result == "[00:00:02] Other: hello"

    def test_empty_segments_list(self) -> None:
        result = format_transcript([])
        assert result == ""

    def test_multiple_speakers(self) -> None:
        segments = [
            Segment(start=0.0, end=3.0, speaker="SPEAKER_00", text="hello"),
            Segment(start=3.5, end=6.0, speaker="You", text="hi there"),
            Segment(start=6.5, end=9.0, speaker="SPEAKER_01", text="welcome"),
        ]
        result = format_transcript(segments)
        lines = result.split("\n")
        assert len(lines) == 3
        assert "SPEAKER_00" in lines[0]
        assert "You" in lines[1]
        assert "SPEAKER_01" in lines[2]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd C:\Users\rafst\documents\projectas\transcription\pipeline
pytest tests/test_formatter.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `pipeline/src/pipeline/formatter.py`:

```python
from pipeline.models import Segment


def format_transcript(segments: list[Segment]) -> str:
    """Format segments into a timestamped, speaker-labeled transcript."""
    sorted_segments = sorted(segments, key=lambda s: s.start)
    lines: list[str] = []
    for seg in sorted_segments:
        if not seg.text.strip():
            continue
        timestamp = _format_time(seg.start)
        lines.append(f"[{timestamp}] {seg.speaker}: {seg.text}")
    return "\n".join(lines)


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd C:\Users\rafst\documents\projectas\transcription\pipeline
pytest tests/test_formatter.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/src/pipeline/formatter.py pipeline/tests/test_formatter.py
git commit -m "feat(pipeline): add transcript formatter with timestamp and speaker labels"
```

---

## Task 8: Pipeline — Audio Preprocessing

**Files:**
- Create: `pipeline/src/pipeline/audio.py`
- Create: `pipeline/tests/conftest.py`
- Test: `pipeline/tests/test_audio.py`

- [ ] **Step 1: Create test fixtures**

Create `pipeline/tests/conftest.py`:

```python
import tempfile

import pytest


@pytest.fixture
def temp_dir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield d
```

- [ ] **Step 2: Write the failing tests**

Create `pipeline/tests/test_audio.py`:

```python
import numpy as np
import numpy.testing as npt
import soundfile as sf
from pathlib import Path
import pytest

from pipeline.audio import split_channels, extract_segment, chunk_audio
from pipeline.exceptions import AudioError


def _make_stereo_wav(path: str, sr: int = 16000, duration: float = 1.0) -> None:
    """Helper: create a 2-channel WAV with distinct tones per channel."""
    samples = int(sr * duration)
    t = np.arange(samples) / sr
    ch0 = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    ch1 = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    sf.write(path, np.column_stack([ch0, ch1]), sr)


class TestSplitChannels:
    def test_should_produce_two_mono_files(self, temp_dir: str) -> None:
        input_path = str(Path(temp_dir) / "stereo.wav")
        _make_stereo_wav(input_path)
        others_path, you_path = split_channels(input_path, temp_dir)
        others, sr0 = sf.read(others_path)
        you, sr1 = sf.read(you_path)
        assert others.ndim == 1
        assert you.ndim == 1
        assert sr0 == 16000
        assert sr1 == 16000

    def test_should_preserve_channel_content(self, temp_dir: str) -> None:
        input_path = str(Path(temp_dir) / "stereo.wav")
        sr = 16000
        samples = sr
        ch0 = np.ones(samples, dtype=np.float32)
        ch1 = np.full(samples, 0.5, dtype=np.float32)
        sf.write(input_path, np.column_stack([ch0, ch1]), sr)

        others_path, you_path = split_channels(input_path, temp_dir)
        others, _ = sf.read(others_path)
        you, _ = sf.read(you_path)
        npt.assert_allclose(others, ch0, atol=1e-6)
        npt.assert_allclose(you, ch1, atol=1e-6)

    def test_should_raise_for_mono_input(self, temp_dir: str) -> None:
        input_path = str(Path(temp_dir) / "mono.wav")
        sf.write(input_path, np.zeros(16000, dtype=np.float32), 16000)
        with pytest.raises(AudioError):
            split_channels(input_path, temp_dir)


class TestExtractSegment:
    def test_should_extract_time_range(self, temp_dir: str) -> None:
        input_path = str(Path(temp_dir) / "full.wav")
        sr = 16000
        audio = np.arange(sr * 4, dtype=np.float32) / (sr * 4)
        sf.write(input_path, audio, sr)

        output_path = str(Path(temp_dir) / "seg.wav")
        extract_segment(input_path, 1.0, 3.0, output_path)
        seg, seg_sr = sf.read(output_path)

        assert seg_sr == sr
        assert len(seg) == sr * 2  # 2 seconds


class TestChunkAudio:
    def test_short_audio_returns_single_chunk(self, temp_dir: str) -> None:
        path = str(Path(temp_dir) / "short.wav")
        sf.write(path, np.zeros(16000 * 10, dtype=np.float32), 16000)
        chunks = chunk_audio(path, max_duration=38.0)
        assert len(chunks) == 1
        assert chunks[0][0] == path

    def test_long_audio_returns_multiple_chunks(self, temp_dir: str) -> None:
        path = str(Path(temp_dir) / "long.wav")
        sf.write(path, np.zeros(16000 * 80, dtype=np.float32), 16000)
        chunks = chunk_audio(path, max_duration=38.0)
        assert len(chunks) == 3  # 38 + 38 + 4 seconds
        for chunk_path, start, end in chunks:
            assert Path(chunk_path).exists()
            assert end > start
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd C:\Users\rafst\documents\projectas\transcription\pipeline
pytest tests/test_audio.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Write the implementation**

Create `pipeline/src/pipeline/audio.py`:

```python
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.exceptions import AudioError


def split_channels(input_path: str, output_dir: str) -> tuple[str, str]:
    """Split a 2-channel WAV into two mono files.

    Returns (others_path, you_path) — channel 0 is others, channel 1 is you.
    """
    audio, sr = sf.read(input_path)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise AudioError(f"Expected 2-channel audio, got shape {audio.shape}")

    others_path = str(Path(output_dir) / "ch0_others.wav")
    you_path = str(Path(output_dir) / "ch1_you.wav")

    sf.write(others_path, audio[:, 0], sr)
    sf.write(you_path, audio[:, 1], sr)

    return others_path, you_path


def extract_segment(
    audio_path: str, start: float, end: float, output_path: str
) -> None:
    """Extract a time range from an audio file and write to output_path."""
    audio, sr = sf.read(audio_path)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment = audio[start_sample:end_sample]
    sf.write(output_path, segment, sr)


def chunk_audio(
    audio_path: str, max_duration: float = 38.0
) -> list[tuple[str, float, float]]:
    """Split audio into chunks under max_duration seconds.

    Returns list of (chunk_path, start_sec, end_sec).
    Short files are returned as-is (no copy).
    """
    audio, sr = sf.read(audio_path)
    total_duration = len(audio) / sr

    if total_duration <= max_duration:
        return [(audio_path, 0.0, total_duration)]

    chunks: list[tuple[str, float, float]] = []
    chunk_samples = int(max_duration * sr)
    parent = Path(audio_path).parent

    for i, start in enumerate(range(0, len(audio), chunk_samples)):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        chunk_path = str(parent / f"chunk_{i:04d}.wav")
        sf.write(chunk_path, chunk, sr)
        chunks.append((chunk_path, start / sr, end / sr))

    return chunks
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd C:\Users\rafst\documents\projectas\transcription\pipeline
pytest tests/test_audio.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline/src/pipeline/audio.py pipeline/tests/conftest.py pipeline/tests/test_audio.py
git commit -m "feat(pipeline): add audio channel splitting, segment extraction, and chunking"
```

---

## Task 9: Pipeline — VAD and Diarization Wrappers

**Files:**
- Create: `pipeline/src/pipeline/vad.py`
- Create: `pipeline/src/pipeline/diarizer.py`

These wrap external models with significant hardware dependencies. Unit tests use mocks; real validation happens in integration testing (Task 12).

- [ ] **Step 1: Write Silero VAD wrapper**

Create `pipeline/src/pipeline/vad.py`:

```python
from typing import Any

import torch

from pipeline.models import Segment


class SileroVAD:
    """Lightweight VAD using Silero (~2MB model). Used for the 'You' channel."""

    def __init__(self) -> None:
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self._get_speech_timestamps: Any = self.utils[0]
        self._read_audio: Any = self.utils[2]

    def detect_speech(
        self, audio_path: str, speaker_label: str = "Speaker"
    ) -> list[Segment]:
        """Detect speech regions and return labeled segments (no text yet)."""
        wav = self._read_audio(audio_path, sampling_rate=16000)
        timestamps = self._get_speech_timestamps(
            wav, self.model, sampling_rate=16000
        )
        segments: list[Segment] = []
        for ts in timestamps:
            segments.append(
                Segment(
                    start=ts["start"] / 16000,
                    end=ts["end"] / 16000,
                    speaker=speaker_label,
                    text="",
                )
            )
        return segments
```

- [ ] **Step 2: Write pyannote diarization wrapper**

Create `pipeline/src/pipeline/diarizer.py`:

```python
import torch
from pyannote.audio import Pipeline as PyannotePipeline

from pipeline.models import Segment


class SpeakerDiarizer:
    """Speaker diarization using pyannote 3.1. Used for the 'Others' channel.

    Requires a HuggingFace token with access to:
    - pyannote/speaker-diarization-3.1
    - pyannote/segmentation-3.0
    Accept user agreements on both model pages first.
    """

    def __init__(self, hf_token: str) -> None:
        self.pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

    def diarize(self, audio_path: str) -> list[Segment]:
        """Run diarization and return speaker-labeled segments (no text yet)."""
        result = self.pipeline(audio_path)
        segments: list[Segment] = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            segments.append(
                Segment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                    text="",
                )
            )
        return segments

    def unload(self) -> None:
        """Free GPU memory for sequential model loading."""
        del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

- [ ] **Step 3: Commit**

```bash
git add pipeline/src/pipeline/vad.py pipeline/src/pipeline/diarizer.py
git commit -m "feat(pipeline): add Silero VAD and pyannote diarization wrappers"
```

---

## Task 10: Pipeline — ASR Transcription Wrapper

**Files:**
- Create: `pipeline/src/pipeline/transcriber.py`

- [ ] **Step 1: Write the Canary SALM transcription wrapper**

Create `pipeline/src/pipeline/transcriber.py`:

```python
from pathlib import Path

import torch
import soundfile as sf
from nemo.collections.speechlm2.models import SALM

from pipeline.audio import extract_segment, chunk_audio
from pipeline.models import Segment


class ASRTranscriber:
    """ASR using NVIDIA Canary QWEN 2.5B via NeMo SALM.

    Uses model.generate() with chat-style prompts. Max 40 seconds per chunk.
    """

    def __init__(self) -> None:
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
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

Note: The exact NeMo SALM API (`model.generate()`, `model.audio_locator_tag`, `model.tokenizer.ids_to_text()`) is based on the current model card. Verify against the actual NeMo version installed on the Jetson — the API may evolve. If `SALM` is not available, fall back to `nemo.collections.asr.models.EncDecMultiTaskModel` with `model.transcribe()`.

- [ ] **Step 2: Commit**

```bash
git add pipeline/src/pipeline/transcriber.py
git commit -m "feat(pipeline): add Canary QWEN ASR transcriber with chunking support"
```

---

## Task 11: Pipeline — Orchestrator and CLI

**Files:**
- Create: `pipeline/src/pipeline/orchestrator.py`
- Create: `pipeline/src/pipeline/cli.py`
- Test: `pipeline/tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing orchestrator test**

Create `pipeline/tests/test_orchestrator.py`:

```python
from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.models import Segment
from pipeline.orchestrator import process_recording
from pipeline.config import PipelineConfig


class TestProcessRecording:
    def _make_test_recording(self, path: str) -> None:
        sr = 16000
        duration = 5.0
        samples = int(sr * duration)
        ch0 = np.random.randn(samples).astype(np.float32) * 0.1
        ch1 = np.random.randn(samples).astype(np.float32) * 0.1
        sf.write(path, np.column_stack([ch0, ch1]), sr)

    @patch("pipeline.orchestrator.ASRTranscriber")
    @patch("pipeline.orchestrator.SileroVAD")
    def test_should_produce_transcript_without_diarization(
        self,
        mock_vad_cls: MagicMock,
        mock_asr_cls: MagicMock,
        temp_dir: str,
    ) -> None:
        input_path = str(Path(temp_dir) / "test.wav")
        output_path = str(Path(temp_dir) / "transcript.txt")
        self._make_test_recording(input_path)

        mock_vad = mock_vad_cls.return_value
        mock_vad.detect_speech.return_value = [
            Segment(start=0.0, end=2.0, speaker="placeholder", text=""),
        ]

        mock_asr = mock_asr_cls.return_value
        mock_asr.transcribe_segments.side_effect = lambda path, segs, td: [
            Segment(start=s.start, end=s.end, speaker=s.speaker, text="mock text")
            for s in segs
        ]

        config = PipelineConfig(enable_diarization=False)
        result = process_recording(input_path, output_path, config)

        assert "mock text" in result
        assert Path(output_path).exists()
        assert "You" in Path(output_path).read_text()

    @patch("pipeline.orchestrator.ASRTranscriber")
    @patch("pipeline.orchestrator.SpeakerDiarizer")
    @patch("pipeline.orchestrator.SileroVAD")
    def test_should_use_diarizer_when_enabled(
        self,
        mock_vad_cls: MagicMock,
        mock_diarizer_cls: MagicMock,
        mock_asr_cls: MagicMock,
        temp_dir: str,
    ) -> None:
        input_path = str(Path(temp_dir) / "test.wav")
        output_path = str(Path(temp_dir) / "transcript.txt")
        self._make_test_recording(input_path)

        mock_vad = mock_vad_cls.return_value
        mock_vad.detect_speech.return_value = [
            Segment(start=0.0, end=2.0, speaker="placeholder", text=""),
        ]

        mock_diarizer = mock_diarizer_cls.return_value
        mock_diarizer.diarize.return_value = [
            Segment(start=0.5, end=3.0, speaker="SPEAKER_00", text=""),
        ]

        mock_asr = mock_asr_cls.return_value
        mock_asr.transcribe_segments.side_effect = lambda path, segs, td: [
            Segment(start=s.start, end=s.end, speaker=s.speaker, text="said something")
            for s in segs
        ]

        config = PipelineConfig(hf_token="fake-token", enable_diarization=True)
        result = process_recording(input_path, output_path, config)

        mock_diarizer_cls.assert_called_once_with("fake-token")
        mock_diarizer.diarize.assert_called_once()
        mock_diarizer.unload.assert_called_once()
        assert "SPEAKER_00" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:\Users\rafst\documents\projectas\transcription\pipeline
pytest tests/test_orchestrator.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the orchestrator**

Create `pipeline/src/pipeline/orchestrator.py`:

```python
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
        you_segments = vad.detect_speech(you_path, speaker_label="You")

        # Step 3: Segment "Others" channel
        if config.enable_diarization and config.hf_token:
            diarizer = SpeakerDiarizer(config.hf_token)
            others_segments = diarizer.diarize(others_path)
            diarizer.unload()
        else:
            others_segments = vad.detect_speech(others_path, speaker_label="Other")

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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd C:\Users\rafst\documents\projectas\transcription\pipeline
pytest tests/test_orchestrator.py -v
```

Expected: all PASS

- [ ] **Step 5: Write the CLI**

Create `pipeline/src/pipeline/cli.py`:

```python
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
```

- [ ] **Step 6: Run all pipeline tests**

```bash
cd C:\Users\rafst\documents\projectas\transcription\pipeline
pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add pipeline/src/pipeline/orchestrator.py pipeline/src/pipeline/cli.py pipeline/tests/test_orchestrator.py
git commit -m "feat(pipeline): add orchestrator with sequential model loading and CLI"
```

---

## Task 12: File Transfer Script

**Files:**
- Create: `scripts/transfer.sh`

- [ ] **Step 1: Write the transfer script**

Create `scripts/transfer.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Transfer a recording to the Jetson for transcription.
# Configure via environment variables or defaults.
#
# Usage:
#   ./scripts/transfer.sh recording_20260328_143000.wav
#   JETSON_HOST=100.x.x.x ./scripts/transfer.sh recording.wav
#
# With Tailscale, JETSON_HOST is typically the Tailscale hostname or IP.

JETSON_HOST="${JETSON_HOST:-jetson}"
JETSON_USER="${JETSON_USER:-user}"
JETSON_DIR="${JETSON_DIR:-~/recordings/inbox}"

if [ $# -eq 0 ]; then
    echo "Usage: transfer.sh <recording.wav> [recording2.wav ...]"
    echo ""
    echo "Environment variables:"
    echo "  JETSON_HOST  Hostname or IP (default: jetson)"
    echo "  JETSON_USER  SSH user (default: user)"
    echo "  JETSON_DIR   Remote directory (default: ~/recordings/inbox)"
    exit 1
fi

# Ensure remote directory exists
ssh "${JETSON_USER}@${JETSON_HOST}" "mkdir -p ${JETSON_DIR}"

for FILE in "$@"; do
    if [ ! -f "$FILE" ]; then
        echo "Error: $FILE not found"
        exit 1
    fi
    BASENAME=$(basename "$FILE")
    SIZE=$(du -h "$FILE" | cut -f1)
    echo "Transferring ${BASENAME} (${SIZE}) to ${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/"
    scp "$FILE" "${JETSON_USER}@${JETSON_HOST}:${JETSON_DIR}/${BASENAME}"
done

echo "Done. Run on Jetson:"
echo "  transcribe ${JETSON_DIR}/<filename>.wav"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/transfer.sh
git add scripts/transfer.sh
git commit -m "feat: add file transfer script for laptop-to-Jetson SCP"
```

---

## Task 13: End-to-End Integration Test

This task validates the full chain on actual hardware. It cannot be automated in CI — run manually.

- [ ] **Step 1: Test the recorder on the Windows laptop**

```bash
cd C:\Users\rafst\documents\projectas\transcription\recorder
pip install -e .

# List devices to confirm loopback + mic are detected
record --list-devices

# Play a YouTube video or any audio, speak into mic, record for 15-20 seconds
record --output-dir ../test-recordings
```

Verify: the output WAV exists, is 2-channel, 16kHz. Play each channel separately to confirm channel 0 has system audio and channel 1 has your voice:

```python
import soundfile as sf
audio, sr = sf.read("test-recordings/<file>.wav")
print(f"Sample rate: {sr}, Channels: {audio.shape[1]}, Duration: {len(audio)/sr:.1f}s")
sf.write("test-recordings/ch0_check.wav", audio[:, 0], sr)
sf.write("test-recordings/ch1_check.wav", audio[:, 1], sr)
# Play ch0_check.wav and ch1_check.wav separately
```

- [ ] **Step 2: Test the pipeline on the Jetson (no diarization first)**

Transfer the test recording to the Jetson, then:

```bash
cd ~/transcription/pipeline
pip install -e .

# Start without diarization (VAD only, lighter)
transcribe ~/recordings/test.wav --no-diarize -o ~/transcripts/test_transcript.txt
```

Verify: transcript has `[HH:MM:SS] You: ...` and `[HH:MM:SS] Other: ...` lines with actual transcribed text.

- [ ] **Step 3: Test the pipeline with diarization**

```bash
# Set HuggingFace token (must have accepted pyannote model agreements)
export HF_TOKEN="hf_your_token_here"

transcribe ~/recordings/test.wav -o ~/transcripts/test_diarized.txt
```

Verify: transcript has distinct speaker labels (e.g., `SPEAKER_00`, `SPEAKER_01`) for the others' channel.

- [ ] **Step 4: Tag the initial version**

```bash
git tag -a v0.1.0 -m "v0.1.0: initial recorder + pipeline with VAD and diarization"
git push origin v0.1.0
```

---

## Verification Checklist

After all tasks are complete, verify end-to-end:

1. **Recorder on laptop:** `record --output-dir ./recordings` captures both system audio and mic into a 2-ch 16kHz WAV
2. **Transfer:** `./scripts/transfer.sh recordings/<file>.wav` sends file to Jetson via SCP
3. **Pipeline on Jetson (VAD mode):** `transcribe <file>.wav --no-diarize` produces a readable transcript with "You" and "Other" labels
4. **Pipeline on Jetson (diarization mode):** `transcribe <file>.wav` produces a transcript with distinct speaker labels on the others' channel
5. **Sequential loading:** Monitor GPU memory during pipeline run — peak should stay under ~5GB (verify with `tegrastats` on Jetson)
6. **All unit tests pass:** `pytest` in both `recorder/` and `pipeline/` directories
