# Understanding LLM Quantization

Quantization is one of the most useful techniques for running Large Language Models (LLMs) on consumer hardware.

It reduces the memory required to store model weights by representing numerical values with fewer bits.

With appropriate quantization, models that would otherwise require substantial GPU memory can become practical to run on systems with limited RAM or VRAM.

This guide explains how quantization works, common formats, GGUF variants, model conversion, and how to choose the right configuration for local inference.

## 1. What Is Quantization?

During training, language models learn numerical weights that determine how they process information and generate responses.

These weights are commonly stored using formats such as FP32, FP16, or BF16.

Quantization converts model weights into lower-precision representations.

For example:

- FP32 uses 32 bits per value.
- FP16 uses 16 bits per value.
- INT8 uses 8 bits per value.
- INT4 uses approximately 4 bits per value.

Reducing precision can significantly reduce model storage and memory requirements.

However, quantization can introduce numerical errors that may affect model quality.

The goal is to find a practical balance between:

- Model quality.
- Memory consumption.
- Inference speed.
- Hardware compatibility.

## 2. Why Quantization Matters

Consider a model with 7 billion parameters.

Its approximate weight-storage requirements are:

| Precision | Approximate weight memory |
|---|---:|
| FP32 | 28 GB |
| FP16 | 14 GB |
| INT8 | 7 GB |
| INT4 | 3.5 GB |

These are theoretical estimates based on parameter count and bits per parameter.

Real model files are usually larger because of metadata, scales, zero points, quantization blocks, and other storage overhead.

Runtime memory requirements are also higher because inference needs additional memory for the KV cache, computation buffers, and other operations.

Quantization makes it possible to experiment with larger models using more modest hardware.

## 3. Common Numerical Formats

### FP32

FP32 represents each value using 32 bits.

Characteristics:

- High numerical precision.
- Large memory requirements.
- Common in some training and scientific computing workflows.

FP32 is generally unnecessary for ordinary consumer local inference.

### FP16

FP16 uses 16-bit floating-point values.

Characteristics:

- Half the weight storage of FP32.
- Widely supported by modern GPUs.
- Useful for inference and training workflows.

FP16 is a common choice when sufficient GPU memory is available.

### BF16

BF16 is another 16-bit floating-point format.

Characteristics:

- Similar memory requirements to FP16.
- Wider numerical range than FP16.
- Useful in many modern machine-learning training workflows.

Hardware and software support vary.

### INT8

INT8 represents values using 8-bit integers.

Characteristics:

- Lower memory requirements than FP16.
- Often maintains good quality.
- Can be useful for inference on supported hardware.

Actual speed improvements depend on the runtime and hardware.

### INT4

INT4 represents values using approximately four bits per value.

Characteristics:

- Substantially reduced weight storage.
- Useful for memory-constrained systems.
- Can introduce greater quality loss than higher-precision formats.

Many popular local LLM quantization methods use block-wise quantization rather than simply storing every weight as a four-bit integer.

## 4. Understanding GGUF

GGUF stands for GPT-Generated Unified Format.

It is a model file format used by llama.cpp and many compatible inference applications.

GGUF stores model weights along with metadata and other information needed by supported inference runtimes.

It is commonly used with:

- llama.cpp.
- LM Studio.
- Ollama.
- Other llama.cpp-compatible applications.

**Important:** GGUF is a file format, not a quantization level.

A GGUF file may contain FP16 weights or different quantized representations.

Example filenames:

```text
model-f16.gguf
model-Q4_K_M.gguf
model-Q5_K_M.gguf
model-Q8_0.gguf
```

The filename generally indicates the type of weight representation used.

## 5. Understanding GGUF Quantization Types

GGUF supports multiple quantization types.

Common examples include:

| Quantization | General characteristics |
|---|---|
| Q2_K | Very small, significant quality trade-offs |
| Q3_K_S | Small, lower precision |
| Q3_K_M | Small, moderate quality trade-offs |
| Q4_0 | Basic four-bit quantization |
| Q4_K_M | Popular balance of size and quality |
| Q5_K_M | Higher precision than Q4_K_M |
| Q6_K | Higher precision, larger storage |
| Q8_0 | High precision, relatively large storage |
| F16 | Half-precision floating-point weights |

The exact variants available depend on the model architecture and llama.cpp version.

### What does K mean?

K-quants are block-based quantization methods used by llama.cpp.

They use different approaches to represent weights efficiently while attempting to preserve useful model quality.

The suffixes also matter:

- S generally indicates a smaller variant.
- M generally indicates a medium variant.

These labels are specific to quantization families, not universal quality ratings.

### Which variant should you choose?

A practical starting point:

- Q4_K_M: A useful general-purpose choice for limited memory.
- Q5_K_M: Consider when you have additional memory.
- Q6_K: Consider when quality is more important than file size.
- Q8_0: Useful when you want relatively high weight precision.
- F16: Useful when preserving half-precision weights is important.

No quantization type is universally best. Evaluate actual model performance for your workload.

## 6. Estimate Quantized Model Sizes

The following estimates provide a rough starting point for common model sizes.

| Model | Approximate Q4 storage | Approximate Q8 storage |
|---|---:|---:|
| 0.5B | 0.3 GB | 0.6 GB |
| 1.5B | 0.9 GB | 1.6 GB |
| 3B | 1.8 GB | 3.2 GB |
| 7B | 4.2 GB | 7.5 GB |
| 8B | 4.8 GB | 8.5 GB |
| 14B | 8.5 GB | 15 GB |
| 32B | 19 GB | 34 GB |
| 70B | 42 GB | 75 GB |

These are illustrative estimates, not guaranteed file sizes.

Actual sizes depend on architecture, quantization implementation, tensor types, and model metadata.

A model's total runtime memory is higher than its file size.

For hardware planning, see the [Hardware Guide](hardware-guide.md).

## 7. Quality vs Memory vs Speed

Quantization introduces trade-offs.

| Approach | Memory | Quality | Speed |
|---|---|---|---|
| FP16 | High | High | Hardware-dependent |
| Q8_0 | Medium-high | Generally high | Hardware-dependent |
| Q6_K | Medium | Generally high | Hardware-dependent |
| Q5_K_M | Lower | Generally good | Hardware-dependent |
| Q4_K_M | Low | Often a good balance | Hardware-dependent |
| Q3 and Q2 variants | Very low | Greater quality trade-offs | Hardware-dependent |

These are broad comparisons, not benchmark results.

Quantization does not automatically make a model faster.

Inference speed depends on:

- CPU and GPU architecture.
- Memory bandwidth.
- GPU offloading.
- Runtime implementation.
- Context length.
- Batch size.
- Quantization kernel support.

A smaller quantized model may run faster on one machine and slower on another.

## 8. CPU and GPU Inference

Quantization can be useful whether you run models on a CPU, GPU, or a combination of both.

### CPU inference

CPU inference can be useful when:

- You do not have a dedicated GPU.
- Your system has sufficient RAM.
- You want to experiment with smaller models.
- Your inference workload does not require high throughput.

llama.cpp supports CPU inference and multiple optimized CPU backends.

### GPU inference

GPU inference can provide better performance when:

- Your GPU supports the required operations.
- Sufficient VRAM is available.
- The inference runtime supports your GPU.
- The model can be loaded fully or partially onto the GPU.

### Partial GPU offloading

Some inference engines allow model layers to be split between GPU memory and system RAM.

This can make larger models runnable on systems with limited VRAM.

However, moving data between system memory and GPU memory can reduce performance.

## 9. Quantize a Model Using llama.cpp

llama.cpp provides tools for converting compatible Hugging Face models into GGUF and quantizing GGUF files.

Repository:

https://github.com/ggml-org/llama.cpp

### Step 1: Install prerequisites

You need:

- Python.
- Git.
- A compatible Hugging Face model.
- llama.cpp source code.
- Sufficient storage space.

Clone the repository:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

Install Python requirements:

```powershell
python -m pip install -r requirements.txt
```

The required Python packages and conversion support may change between releases.

### Step 2: Obtain a compatible model

Download a compatible Hugging Face model or clone its repository.

For example:

```powershell
git lfs install
git clone https://huggingface.co/ORG/MODEL
```

Replace `ORG/MODEL` with the actual model repository.

Some repositories require authentication or acceptance of specific usage terms.

Make sure you have enough disk space for the original model and converted files.

### Step 3: Convert the model to GGUF

The following example converts a compatible Hugging Face model into an FP16 GGUF file.

```powershell
python .\convert_hf_to_gguf.py .\MODEL `
  --outfile .\model-f16.gguf `
  --outtype f16
```

Replace `MODEL` with the local model directory.

Conversion support depends on the model architecture and the current llama.cpp implementation.

### Step 4: Build llama.cpp quantization tools

On Windows, install the required CMake and C++ build tools.

From the llama.cpp directory:

```powershell
cmake -B build
cmake --build build --config Release
```

The resulting executable is commonly located at:

```text
build\bin\Release\llama-quantize.exe
```

Exact build paths can vary depending on the generator and build configuration.

### Step 5: Quantize the GGUF model

Example using Q4_K_M:

```powershell
.\build\bin\Release\llama-quantize.exe `
  .\model-f16.gguf `
  .\model-Q4_K_M.gguf `
  Q4_K_M
```

You can substitute another supported quantization type.

For example:

```text
Q5_K_M
Q6_K
Q8_0
```

Check the available quantization types in the llama.cpp documentation for your version.

### Step 6: Compare file sizes

Use PowerShell:

```powershell
Get-ChildItem *.gguf |
  Select-Object Name, Length
```

Compare the original FP16 file with the quantized version.

A smaller file generally means lower weight-storage requirements.

It does not automatically mean better performance.

## 10. Run a GGUF Model Using Ollama

Ollama supports importing compatible GGUF models.

After creating your quantized GGUF file, create a file named `Modelfile`:

```text
FROM ./model-Q4_K_M.gguf
```

Then create the Ollama model:

```powershell
ollama create my-local-model -f .\Modelfile
```

Run it:

```powershell
ollama run my-local-model
```

Check available models:

```powershell
ollama list
```

Remove the model when it is no longer needed:

```powershell
ollama rm my-local-model
```

See the official documentation:

https://docs.ollama.com/import

## 11. Compare Quantization Variants

A useful experiment is to compare multiple quantization variants of the same model.

For example:

```text
model-Q4_K_M.gguf
model-Q5_K_M.gguf
model-Q6_K.gguf
model-Q8_0.gguf
```

Use identical:

- Model architecture and original checkpoint.
- Prompts.
- Context settings.
- Hardware.
- Inference runtime.
- Generation settings.

Measure:

1. Model file size.
2. Peak RAM usage.
3. Peak VRAM usage.
4. Time to first token.
5. Tokens generated per second.
6. Response quality.

Use several representative prompts rather than judging a model from one answer.

This helps you identify the best practical configuration for your hardware.

## 12. Common Quantization Mistakes

### Mistake 1: Assuming file size equals memory usage

A 4 GB model does not necessarily require only 4 GB of system memory.

Inference needs additional memory for the runtime, KV cache, and other operations.

### Mistake 2: Choosing the smallest quantization automatically

Smaller files can come with quality trade-offs.

Choose based on actual task performance, not file size alone.

### Mistake 3: Confusing GGUF with Q4

GGUF is a file format.

Q4_K_M is a quantization type that can be stored in a GGUF file.

### Mistake 4: Ignoring context length

Longer context windows can increase KV cache memory usage.

A model that runs comfortably at a short context length may require substantially more memory at a longer context.

### Mistake 5: Assuming quantization always increases speed

Quantization can reduce memory bandwidth requirements, but actual performance depends on hardware and optimized kernels.

### Mistake 6: Downloading untrusted model conversions

Prefer official model repositories or reputable quantized versions.

Check model provenance, conversion details, and licensing before using third-party files.

## 13. Practical Recommendations

For beginners:

- Start with a compatible Q4_K_M model.
- Use Ollama or LM Studio to simplify local inference.
- Monitor memory consumption.
- Compare Q4 and Q5 if your hardware allows.
- Avoid selecting a model solely by its advertised parameter count.

For developers:

- Learn the llama.cpp conversion and quantization workflow.
- Keep the original model checkpoint when possible.
- Record the model revision and quantization settings.
- Benchmark multiple variants.
- Document the hardware and inference configuration used.

For researchers:

- Evaluate quality degradation on representative tasks.
- Compare quantization methods consistently.
- Record the original checkpoint and conversion tooling.
- Consider calibration data and importance-aware quantization when appropriate.

## 14. Further Reading

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Ollama Documentation](https://docs.ollama.com/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [GGUF Format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Getting Started with Local LLMs](getting-started.md)
- [Hardware Guide](hardware-guide.md)
- [Model Selection Guide](model-selection.md)

---

**Remember:** Quantization is about making models practical for your hardware while preserving as much useful quality as possible. The best choice is the one that works reliably for your actual workload.