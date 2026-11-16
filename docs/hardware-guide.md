# Hardware Guide for Local LLMs

Running Large Language Models (LLMs) locally does not always require an expensive GPU or workstation.

However, hardware determines which models you can run, how quickly they respond, and whether you can fine-tune them.

This guide explains the hardware requirements for local LLM inference and fine-tuning, with practical recommendations for different hardware configurations.

## 1. Understanding the Main Hardware Components

Before choosing a model, understand the components that affect local AI performance.

### RAM (System Memory)

RAM is your computer's main working memory.

It is used by:
- The operating system.
- Running applications.
- CPU-based model inference.
- Model weights and other inference data when loaded into system memory.

More RAM allows you to work with larger models and run more applications simultaneously.

### VRAM (Video Memory)

VRAM is dedicated memory available on a GPU.

It is commonly used for:
- Loading model weights onto the GPU.
- Processing input tokens.
- Storing intermediate computation data.
- Managing the model's context and KV cache.

GPU memory is often the main limitation when running larger models entirely on a graphics card.

### CPU (Processor)

The CPU can run local LLM inference even without a dedicated GPU.

CPU inference is useful when:
- Your computer does not have a supported GPU.
- You want to experiment with smaller models.
- You have sufficient system RAM.

However, CPU inference is generally slower than optimized GPU inference.

### GPU (Graphics Processing Unit)

A compatible GPU can significantly accelerate model inference.

GPU performance depends on:
- VRAM capacity.
- Memory bandwidth.
- Compute capability.
- Software support.
- Model architecture and quantization.

A GPU with more VRAM is not automatically faster than every GPU with less VRAM.

## 2. RAM vs VRAM

RAM and VRAM serve different purposes.

| Feature | RAM | VRAM |
|---|---|---|
| Primary use | System memory | GPU memory |
| Used by | CPU and applications | GPU workloads |
| Model inference | CPU and shared/offloaded workloads | GPU-resident workloads |
| Upgradeability | Often upgradeable | Usually fixed on discrete GPUs |
| Main limitation | System memory capacity | GPU memory capacity |

Some systems use unified memory or shared GPU memory. This can change how memory is allocated, but shared memory should not be assumed to perform like dedicated VRAM.

## 3. Understanding Model Size and Memory

Model size is commonly expressed in parameters.

For example:
- 0.5B means approximately 500 million parameters.
- 1.5B means approximately 1.5 billion parameters.
- 3B means approximately 3 billion parameters.
- 7B means approximately 7 billion parameters.
- 14B means approximately 14 billion parameters.

Larger models generally require more memory, although architecture, precision, quantization, and runtime overhead also matter.

### Approximate Weight Memory

The following table estimates memory required for model weights alone.

| Model size | FP16/BF16 | 8-bit | 4-bit |
|---|---:|---:|---:|
| 0.5B | 1 GB | 0.5 GB | 0.25–0.4 GB |
| 1.5B | 3 GB | 1.5 GB | 0.75–1 GB |
| 3B | 6 GB | 3 GB | 1.5–2 GB |
| 7B | 14 GB | 7 GB | 3.5–4.5 GB |
| 8B | 16 GB | 8 GB | 4–5 GB |
| 14B | 28 GB | 14 GB | 7–8 GB |

These are rough estimates, not guaranteed runtime requirements.

Actual memory consumption is higher because inference also requires:
- KV cache.
- Activations and temporary buffers.
- Runtime overhead.
- Context-dependent memory.
- Other GPU and system processes.

## 4. GPU VRAM Recommendations

The following recommendations are starting points for experimentation, not strict compatibility guarantees.

| Dedicated VRAM | Practical starting point |
|---|---|
| 2–4 GB | Small quantized models, approximately 0.5B–1.5B |
| 6 GB | Small models and some quantized 3B models |
| 8 GB | Quantized 3B models and selected 7B models with careful settings |
| 12 GB | Many quantized 7B–8B models |
| 16 GB | More flexibility with 7B–8B models and selected larger workloads |
| 24 GB | Larger quantized models and more demanding experiments |

The exact model, context length, runtime, and available memory determine what actually fits.

A model that partially fits in VRAM may still run through CPU offloading, but performance can be considerably different.

## 5. Running LLMs Without a Dedicated GPU

A dedicated GPU is not mandatory for local LLM inference.

You can use:
- CPU inference with llama.cpp.
- Ollama with supported CPU execution.
- LM Studio with CPU inference options.

### Advantages

- No dedicated GPU required.
- Useful for learning and experimentation.
- Can work with existing computers.
- Avoids GPU purchase costs.

### Limitations

- Generation can be slower.
- Larger models require substantial RAM.
- Performance depends heavily on CPU architecture and memory bandwidth.

For beginners, a small quantized model is a sensible starting point.

## 6. Hardware for Fine-Tuning

Fine-tuning has different hardware requirements from ordinary inference.

Inference generates responses using an existing model. Fine-tuning updates model parameters or trains additional parameters to adapt a model to a particular task.

### Common fine-tuning approaches

**Full Fine-Tuning**

Updates all or most model parameters. It can require substantial GPU memory for weights, gradients, optimizer states, and activations.

**LoRA**

Low-Rank Adaptation trains smaller additional parameter matrices while keeping the original model weights frozen.

**QLoRA**

Quantized Low-Rank Adaptation loads a quantized base model and trains LoRA adapters. It can reduce memory requirements significantly compared with full fine-tuning.

### Fine-Tuning Hardware Considerations

| Hardware | Practical guidance |
|---|---|
| 4 GB VRAM | Very limited; suitable for experimentation with constrained setups, not general-purpose fine-tuning |
| 6 GB VRAM | Small-model experiments with aggressive memory optimization |
| 8 GB VRAM | Some small-model QLoRA workflows may be possible |
| 12 GB VRAM | More flexibility for small-model fine-tuning |
| 16 GB VRAM | Better headroom for selected QLoRA workloads |
| 24 GB VRAM | More practical for larger fine-tuning experiments |

Actual feasibility depends on model architecture, sequence length, batch size, optimizer, gradient checkpointing, and training framework.

Tools worth exploring:
- Unsloth
- LLaMA-Factory
- Hugging Face Transformers
- PEFT

## 7. Inference vs Fine-Tuning vs Training from Scratch

These are three different workloads.

| Workload | Purpose | Relative hardware demand |
|---|---|---|
| Inference | Run an existing model | Low to high |
| Fine-tuning | Adapt an existing model | High |
| Training from scratch | Train a new model | Extremely high |

For most individuals and small teams, local inference and parameter-efficient fine-tuning are the most practical starting points.

Training a competitive foundation model from scratch generally requires substantial compute infrastructure and large datasets.

## 8. Choosing Hardware Based on Your Budget

Rather than purchasing hardware immediately, evaluate what you already own.

### Entry-Level Setup

Suitable for:
- Learning local inference.
- Running small quantized models.
- Experimenting with Ollama.

Suggested starting point:
- 8–16 GB system RAM.
- Modern multi-core CPU.
- SSD storage.
- No dedicated GPU required.

### Mid-Range Setup

Suitable for:
- Running quantized 7B–8B models.
- Faster inference.
- Selected small-model fine-tuning experiments.

Suggested starting point:
- 16–32 GB system RAM.
- GPU with 8–12 GB VRAM.
- SSD storage.

### Advanced Setup

Suitable for:
- Larger local models.
- More demanding inference workloads.
- Selected QLoRA fine-tuning workflows.

Suggested starting point:
- 32–64 GB or more system RAM.
- GPU with 16–24 GB or more VRAM.
- Fast SSD storage.

These are planning categories rather than guaranteed performance targets.

## 9. Storage Requirements

Models can consume several gigabytes of disk space.

For example:
- Small quantized models may occupy less than 1 GB.
- Quantized 7B–8B models commonly occupy several gigabytes.
- Larger models can require tens of gigabytes or more.

Leave additional space for:
- Multiple model versions.
- Downloaded datasets.
- Training checkpoints.
- Model conversions.
- Application dependencies.

An SSD is recommended for faster model loading and general workflow performance.

## 10. How to Check Your Hardware

### Windows

Open PowerShell:

```powershell
Get-CimInstance Win32_ComputerSystem |
    Select-Object Manufacturer, Model, TotalPhysicalMemory

Get-CimInstance Win32_Processor |
    Select-Object Name, NumberOfCores, NumberOfLogicalProcessors

Get-CimInstance Win32_VideoController |
    Select-Object Name, AdapterRAM
```

For NVIDIA GPUs, check:

```powershell
nvidia-smi
```

Note that `AdapterRAM` reported by Windows can be incomplete or misleading for some graphics devices. Use the GPU manufacturer's tools for more reliable memory information.

### Linux

Check system memory:

```bash
free -h
```

Check CPU:

```bash
lscpu
```

Check NVIDIA GPU:

```bash
nvidia-smi
```

Check disk space:

```bash
df -h
```

### macOS

Check system information:

```bash
system_profiler SPHardwareDataType
```

Check available storage:

```bash
df -h
```

Apple Silicon systems use unified memory, which is shared between the CPU and GPU.

## 11. Practical Recommendations

Before downloading a model:

1. Check your available RAM and VRAM.
2. Identify the model's parameter count.
3. Check its precision and quantization.
4. Account for context length and KV cache.
5. Leave memory available for your operating system.
6. Start with a smaller model and measure actual performance.

Do not assume that a model will run well just because its weight file is smaller than your available VRAM.

## 12. What's Next?

Continue exploring:

- [Getting Started](getting-started.md)
- [Model Selection](model-selection.md)
- [Quantization](quantization.md)
- [Troubleshooting](troubleshooting.md)

Explore practical workflows in the `guides/` directory.

---

**Local LLM Cookbook** — Learn, experiment, and build with AI models running on your own hardware.