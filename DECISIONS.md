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
