# Local Transcription Setup — Jetson Orin Nano

## ASR Model

- **ID:** `nvidia/canary-qwen-2.5b`
- **Size:** 2.5B parameters
- **Architecture:** NeMo-based ASR + LLM post-processing
- **Language:** English
- **Framework:** NeMo (NVIDIA) — direct path to ONNX → TensorRT
- **Modes:** ASR (speech-to-text with punctuation/capitalization), LLM (transcript summarization, Q&A)

### Why Canary over Cohere Transcribe

Evaluated `CohereLabs/cohere-transcribe-03-2026` (2B params, 14 languages, 5.42% avg WER, Apache 2.0). Cohere has slightly better raw WER benchmarks and multilingual support, but for the Jetson:

- **NVIDIA-on-NVIDIA toolchain**: Canary is a NeMo model with a direct `nemo2onnx` → TensorRT export path, which is NVIDIA's own inference stack built for Jetson
- **Jetson-optimized containers and tooling** exist for NeMo models; Cohere requires HuggingFace Transformers (>=5.4.0) which has weaker Jetson support
- **Quantization**: NVIDIA's INT8/INT4 quantization tooling is more mature for their own models — the WER gap likely narrows or reverses after quantization
- Multilingual support not needed (English-only use case)

## Speaker Diarization

- **Model:** `pyannote/speaker-diarization-3.1`
- **Purpose:** Identify who is speaking when (speaker segmentation + labeling)
- **Memory:** ~1.5 GB quantized on Jetson
- **Note:** Only needed on the "others" audio channel — see two-channel approach below

## Hardware Target: Jetson Orin Nano 8GB

- **GPU:** 1024 CUDA cores, 32 Tensor Cores (Ampere), 1020 MHz
- **CPU:** 6-core Arm Cortex-A78AE @ 1.7 GHz
- **Memory:** 8 GB LPDDR5 (shared CPU/GPU), 102 GB/s bandwidth
- **Power:** 7W / 10W / 15W / 25W configurable
- **Price:** ~$249 (Super Developer Kit)

### Memory Budget

| Component | Estimated Memory |
|-----------|-----------------|
| Canary QWEN 2.5B (INT8) | ~3-4 GB |
| pyannote diarization 3.1 | ~1.5 GB |
| OS + buffers | ~2 GB |
| **Total** | **~6.5-7.5 GB** |

- fp16 model needs ~5 GB for weights alone — too tight with diarization + OS on 8 GB shared memory
- **Must use INT8 or INT4 quantization** for the ASR model
- Use JetPack SDK's prebuilt PyTorch wheels, not vanilla pip
- Use TensorRT optimization via NeMo export pipeline
- Use NVMe SSD, not microSD, for model loading

## Use Case: Zoom Call Transcription (Post-Call)

Personal use — transcribe calls after they end so nothing gets missed. Not live transcription.

### Architecture

```
DURING CALL (laptop only):

┌─── Laptop (Windows 11) ──────────────┐
│                                       │
│  WASAPI Loopback ──┐                  │
│  (others' audio)   ├──→ 2-ch .wav     │
│  WASAPI Mic ───────┘    file          │
│  (your voice)                         │
│                                       │
│  Zoom call runs normally,             │
│  audio heard through speakers/        │
│  headphones as usual                  │
└───────────────────────────────────────┘

AFTER CALL (send to Jetson):

┌─── Laptop ───┐          ┌─── Jetson Orin Nano ──────────┐
│               │          │                               │
│  recording    │  WiFi /  │  Receive .wav file            │
│  .wav ────────┼──────────┼──→ Ch1 (others) → Diarize    │
│               │ Tailscale│      → ASR per speaker        │
│               │          │   Ch2 (you) → ASR             │
│               │          │      (label as "You")         │
│               │          │          │                    │
│               │          │          ▼                    │
│               │          │   transcript.txt              │
└───────────────┘          └───────────────────────────────┘
```

### Two-Channel Recording (Laptop Side)

The laptop runs a Python script that captures two streams simultaneously:

- **Channel 1 — WASAPI Loopback:** Captures a copy of system audio output (other Zoom participants). Does not modify or redirect the audio — you hear everything normally through speakers/headphones.
- **Channel 2 — WASAPI Mic:** Captures your microphone input. Does not interfere with Zoom's mic access.

Output: a single 2-channel WAV file (16kHz, 16-bit). A 1-hour call ≈ 450 MB raw, or ~150 MB as FLAC (lossless compression).

### Transcription Pipeline (Jetson Side)

1. Receive the 2-channel audio file
2. **Channel 2 (your mic):** Transcribe directly with Canary, label all output as "You" — no diarization needed
3. **Channel 1 (others):** Run pyannote diarization to identify speaker segments, then transcribe each segment with Canary
4. Merge both channels into a single chronological transcript with speaker labels
5. Output to file (or simple local web UI)

### Benefits of Batch Over Real-Time

- **No network dependency during calls** — nothing can drop or lag mid-call
- **Batch processing is faster** — Jetson can process chunks in parallel without waiting on an audio stream
- **Simpler code** — recorder is ~50 lines of Python, no socket/streaming management
- **Raw audio preserved** — can re-transcribe if models are upgraded later

## Zoom Detection

**WASAPI loopback is invisible to Zoom.** It operates at the OS level, reading a copy of the audio output buffer. Zoom's "recording" notification is triggered only by Zoom's own in-app recording feature (cloud or local recording through the Zoom UI). An OS-level audio capture:

- Sends no signal to Zoom
- Generates no network traffic to Zoom's servers
- Is indistinguishable from audio simply playing to speakers
- Does not require VB-Cable or virtual audio drivers (since we're not streaming real-time)

## Remote Access (Laptop Away from Jetson)

For when the laptop is not on the same local network as the Jetson:

| Option | How | Tradeoff |
|--------|-----|----------|
| **Tailscale** (recommended) | Virtual private network between devices. Both see each other as if on the same LAN. | Simplest. Free tier sufficient. ~5-15ms added latency (irrelevant for batch file transfer). |
| **ZeroTier** | Same concept as Tailscale, different provider. | Similar ease of setup. |
| **WireGuard VPN** | Jetson runs WireGuard server, laptop connects as client. | No third-party dependency. Needs port forwarding on home router. |
| **SSH tunnel** | `ssh -R` from Jetson to a cheap VPS, laptop connects to VPS. | Works when port forwarding is not possible. Adds a hop. |

Audio file transfer is lightweight (~150 MB FLAC for a 1-hour call) — any of these options handle it easily.

## Limitations

- **English only** — Canary QWEN 2.5B does not support other languages
- **No live transcription** — transcript is produced after the call ends
- **Diarization accuracy** — pyannote may struggle with overlapping speech or very similar voices
- **8 GB memory is tight** — limited headroom for model upgrades or additional processing
- **No timestamps in ASR output** — timestamps come from the diarization segments, not the ASR model itself
