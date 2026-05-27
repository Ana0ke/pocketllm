# PocketLLM 🧊

**Put any LLM in your pocket. One command to quantize and deploy to edge devices.**

```bash
pocketllm Qwen/Qwen2.5-1.5B-Instruct --target rpi5 --quant int4
```

That's it. You get a ready-to-run deployment package optimized for your target device.

## Why PocketLLM?

Running LLMs on edge devices (Raspberry Pi, Jetson, Android phones) is painful:

- **llama.cpp** only does GGUF quantization
- **MNN** only does MNN conversion
- **auto-gptq** only does GPTQ quantization
- Each tool has its own flags, formats, and deployment quirks

PocketLLM unifies them all behind one CLI. Pick a model, pick a device, pick a quantization method — get a deployable package.

## Features

- 🎯 **One command** from HuggingFace model to edge deployment
- 🔧 **Multiple quantization backends**: GGUF (llama.cpp), GPTQ, AWQ, MNN
- 📱 **Multiple target devices**: Raspberry Pi 5, Jetson Nano, Android, generic Linux
- 📊 **Auto benchmark**: speed, memory usage, and accuracy loss report
- 🧠 **Smart defaults**: automatically picks the best quantization for your device
- 📦 **Ready-to-run packages**: includes runtime binaries, configs, and launch scripts

## Quick Start

```bash
# Install
pip install pocketllm

# Quantize and package for Raspberry Pi 5
pocketllm Qwen/Qwen2.5-1.5B-Instruct --target rpi5 --quant int4

# Auto-select best quantization for device
pocketllm Qwen/Qwen2.5-1.5B-Instruct --target rpi5

# List supported devices
pocketllm devices

# List supported quantization methods
pocketllm quants

# Benchmark a quantized model
pocketllm benchmark ./output/Qwen2.5-1.5B-Instruct-rpi5-int4
```

## Supported Models (v0.1)

| Model | Params | GGUF | GPTQ | MNN |
|-------|--------|------|------|-----|
| Qwen2.5-0.5B-Instruct | 0.5B | ✅ | ✅ | ✅ |
| Qwen2.5-1.5B-Instruct | 1.5B | ✅ | ✅ | ✅ |
| Qwen2.5-3B-Instruct | 3B | ✅ | ✅ | ✅ |
| Phi-3-mini-instruct | 3.8B | ✅ | ✅ | ✅ |
| Gemma-2-2B-it | 2B | ✅ | ✅ | ✅ |
| TinyLlama-1.1B-Chat | 1.1B | ✅ | ✅ | ✅ |

## Supported Devices (v0.1)

| Device | RAM | Arch | Backend |
|--------|-----|------|---------|
| Raspberry Pi 5 | 8GB | ARM64 | llama.cpp |
| Jetson Nano | 4GB | ARM64+CUDA | llama.cpp / TensorRT |
| Android (mid-range) | 8GB | ARM64+NPU | MNN |
| Generic Linux | varies | x86_64 | llama.cpp |

## How It Works

```
HuggingFace Model
       │
       ▼
  ┌─────────┐    ┌──────────────┐    ┌──────────┐
  │ Download │───▶│  Quantize    │───▶│  Package  │
  │  Model   │    │ (GGUF/GPTQ/ │    │ (device-  │
  │          │    │  AWQ/MNN)   │    │  specific)│
  └─────────┘    └──────────────┘    └──────────┘
                                           │
                                           ▼
                                    Deployable Package
                                    (.tar.gz / .deb / APK)
```

## Device Profiles

Device profiles are simple YAML files. You can contribute new devices:

```yaml
# devices/profiles/rpi5.yaml
name: Raspberry Pi 5
ram_gb: 8
cpu: ARM Cortex-A76
cpu_cores: 4
arch: aarch64
gpu: none
recommended_quant: int4
max_model_size_gb: 6.5
backends:
  - llama.cpp
  - MNN
```

## Roadmap

- [x] **v0.1** — GGUF quantization + Raspberry Pi 5 support
- [ ] **v0.2** — Android (MNN) + Jetson (TensorRT) targets
- [ ] **v0.3** — Benchmark module + Gradio Web UI
- [ ] **v1.0** — Custom device profiles, LoRA merge pipeline, CI/CD

## Contributing

We welcome contributions! Especially:

- 📱 New device profiles (just add a YAML file)
- 🔧 New quantization backends
- 🧪 Test results from real hardware
- 📖 Documentation and tutorials

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Apache License 2.0 — use it however you want.
