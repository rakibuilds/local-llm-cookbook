# Local LLM Troubleshooting Guide

Running Large Language Models locally can involve several components: model files, inference engines, Python environments, GPU drivers, and system memory.

When something goes wrong, the most effective approach is to identify the failing component before changing your configuration.

This guide covers common problems with Ollama, LM Studio, llama.cpp, Python, GPU acceleration, model downloads, and local inference.

## 1. Start with Basic Diagnostics

Before troubleshooting, collect basic information about your system.

### Windows PowerShell

Check your operating system:

```powershell
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion
```

Check available memory:

```powershell
Get-CimInstance Win32_OperatingSystem |
  Select-Object TotalVisibleMemorySize, FreePhysicalMemory
```

Check your processor:

```powershell
Get-CimInstance Win32_Processor |
  Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
```

Check graphics hardware:

```powershell
Get-CimInstance Win32_VideoController |
  Select-Object Name, AdapterRAM, DriverVersion
```

Note that `AdapterRAM` may not accurately report dedicated GPU memory on every system.

For NVIDIA GPUs, use:

```powershell
nvidia-smi
```

### Linux

Check operating system information:

```bash
uname -a
```

Check memory:

```bash
free -h
```

Check processor information:

```bash
lscpu
```

Check storage:

```bash
df -h
```

Check NVIDIA GPU status:

```bash
nvidia-smi
```

Some commands require additional system packages. NVIDIA-specific commands are only available when the appropriate drivers and utilities are installed.

## 2. Ollama Troubleshooting

Ollama makes local model inference easier, but installation, networking, and model-loading problems can still occur.

### Problem: `ollama` is not recognized

Example:

```text
ollama : The term 'ollama' is not recognized
```

**Possible causes:**

- Ollama is not installed.
- The installation directory is missing from PATH.
- The terminal was opened before installation completed.

**Solutions:**

1. Install Ollama from the official website.
2. Close and reopen PowerShell.
3. Check whether the command is available:

```powershell
ollama --version
```

Official download:

https://ollama.com/download

If the command remains unavailable, verify the installation and PATH configuration.

### Problem: Ollama service is not running

Example:

```text
Error: could not connect to ollama server
```

**Possible causes:**

- Ollama is not running.
- Its background service failed.
- Another process is interfering with the local service.

**Solutions:**

On Windows, start the Ollama application from the Start menu.

On Linux, check the service:

```bash
systemctl status ollama
```

Start it if necessary:

```bash
sudo systemctl start ollama
```

For a manual foreground session, where appropriate:

```bash
ollama serve
```

Do not start a second server if one is already running on the same port.

Check whether the local API responds:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

On Linux:

```bash
curl http://localhost:11434/api/tags
```

### Problem: Model download fails

Example:

```text
Error: pull model manifest: file does not exist
```

**Possible causes:**

- Incorrect model name or tag.
- Model is unavailable.
- Network connectivity problems.
- Registry or download interruption.

**Solutions:**

Check the official model library:

https://ollama.com/library

Then run a valid model name:

```powershell
ollama run qwen3:0.6b
```

The example tag is illustrative; verify current availability in the library.

If a download is interrupted, retry the pull:

```powershell
ollama pull qwen3:0.6b
```

### Problem: Model runs very slowly

**Possible causes:**

- Model is too large for available memory.
- GPU acceleration is unavailable.
- Model is running mostly on the CPU.
- Context length is too large.
- Other applications are consuming resources.

**Solutions:**

1. Try a smaller model.
2. Use a more memory-efficient quantization.
3. Close unnecessary applications.
4. Check GPU and system memory usage.
5. Reduce the context length if appropriate.

Inspect running models:

```powershell
ollama ps
```

List downloaded models:

```powershell
ollama list
```

### Problem: Ollama reports insufficient memory

Example:

```text
model requires more system memory
```

**Possible causes:**

- Model weights exceed available memory.
- KV cache requirements are too high.
- Other applications consume too much RAM or VRAM.
- Requested context length is too large.

**Solutions:**

- Select a smaller model.
- Use a more compact quantized version.
- Reduce context length.
- Close memory-intensive applications.
- Avoid loading multiple models simultaneously.

Remember that model file size is not the same as total runtime memory usage.

### Problem: Local API does not respond

Ollama normally exposes its local API at:

```text
http://localhost:11434
```

Check the API:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

If it fails:

1. Confirm Ollama is running.
2. Check whether port 11434 is listening.
3. Check local firewall or security software settings.
4. Verify that the application is using the correct API address.

Check the port on Windows:

```powershell
Get-NetTCPConnection -LocalPort 11434 -State Listen
```

If the API is exposed beyond localhost, review network access and security carefully. Do not expose an unauthenticated local inference service to an untrusted network.

## 3. LM Studio Troubleshooting

LM Studio provides a graphical interface for downloading and running local models.

### Problem: Model fails to load

**Possible causes:**

- Insufficient RAM or VRAM.
- Unsupported model architecture.
- Incompatible model format.
- Corrupted or incomplete model download.

**Solutions:**

1. Verify that the model is supported by your LM Studio version.
2. Check the model's documentation.
3. Select a smaller or more suitable quantized model.
4. Restart LM Studio.
5. Try another compatible model.

### Problem: Model download is interrupted

**Possible causes:**

- Unstable internet connection.
- Insufficient disk space.
- Download server interruption.

**Solutions:**

- Check available storage.
- Verify your internet connection.
- Retry the download.
- Confirm that the model files are complete.

### Problem: GPU offloading is unavailable

**Possible causes:**

- Unsupported GPU.
- Missing or incompatible GPU drivers.
- Unsupported acceleration backend.
- Model or runtime compatibility issues.

**Solutions:**

1. Update your GPU drivers.
2. Update LM Studio.
3. Check the available hardware acceleration options.
4. Try CPU inference to isolate the problem.
5. Test with a smaller compatible model.

Do not assume that every GPU supports every acceleration backend.

Official website:

https://lmstudio.ai/

## 4. GPU and CUDA Troubleshooting

GPU acceleration can improve inference performance, but it depends on hardware, drivers, and software compatibility.

### Problem: NVIDIA GPU is not detected

Run:

```powershell
nvidia-smi
```

If the command fails, possible causes include:

- NVIDIA driver is missing.
- Driver installation is corrupted.
- GPU is not supported by the installed driver.
- NVIDIA utilities are unavailable in PATH.

**Solutions:**

1. Verify that your computer has a compatible NVIDIA GPU.
2. Install an appropriate driver from NVIDIA.
3. Restart your computer.
4. Run `nvidia-smi` again.

Official driver downloads:

https://www.nvidia.com/Download/index.aspx

### Problem: CUDA is unavailable in Python

Check whether PyTorch detects CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Display GPU information:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

If CUDA is unavailable:

- Verify the installed PyTorch build.
- Check GPU and driver compatibility.
- Confirm that the correct Python environment is active.
- Install a compatible PyTorch version.

Official PyTorch installation instructions:

https://pytorch.org/get-started/locally/

**Important:** CUDA support in PyTorch is separate from GPU support in Ollama, LM Studio, and llama.cpp. One application working with your GPU does not guarantee that another application will work.

## 5. Out-of-Memory Errors

Out-of-memory errors occur when an application requests more memory than the available hardware or configured runtime can provide.

Example:

```text
CUDA out of memory
```

### Common causes

- Model is too large.
- Quantization is too high-precision for available memory.
- Context length is too large.
- Batch size is too large.
- Multiple models are loaded.
- Other applications consume GPU memory.

### Recommended solutions

**1. Use a smaller model**

Try a smaller parameter count.

**2. Choose a more compact quantization**

For example, consider Q4_K_M instead of Q8_0 if the quality trade-off is acceptable.

**3. Reduce context length**

Long context windows can increase KV cache memory consumption.

**4. Reduce batch size**

This can help in applications where batch size is configurable.

**5. Close unnecessary applications**

Free system RAM and GPU memory before loading a model.

**6. Use partial GPU offloading**

Some inference engines support running part of a model on the GPU and the remaining part in system RAM.

This can reduce VRAM requirements but may reduce performance.

## 6. Slow Inference and Poor Performance

A model may load successfully but generate responses very slowly.

### Possible causes

- CPU-only inference.
- Limited memory bandwidth.
- Insufficient VRAM.
- Large model size.
- High context length.
- Thermal throttling.
- Background processes.
- Inefficient quantization or runtime configuration.

### Troubleshooting steps

Check system resource usage.

On Windows:

```powershell
Get-Process |
  Sort-Object CPU -Descending |
  Select-Object -First 10 ProcessName, CPU
```

On Linux:

```bash
top
```

For NVIDIA GPUs:

```bash
nvidia-smi
```

Try the following:

1. Test a smaller model.
2. Compare another quantization variant.
3. Reduce context length.
4. Check GPU utilization and memory usage.
5. Monitor CPU and GPU temperatures.
6. Close unnecessary applications.
7. Update the inference software.

Compare performance using identical prompts and generation settings.

Do not compare tokens-per-second results from different hardware or configurations without documenting the differences.

## 7. GGUF and llama.cpp Troubleshooting

GGUF models are widely used for local inference through llama.cpp-compatible applications.

### Problem: GGUF model fails to load

**Possible causes:**

- Unsupported architecture.
- Corrupted model file.
- Incompatible GGUF version.
- Incompatible inference runtime.
- Incorrect model path.

**Solutions:**

1. Download the model again from a trusted source.
2. Verify the file size and download integrity when a checksum is available.
3. Update llama.cpp.
4. Check whether the model architecture is supported.
5. Confirm the file path.

### Problem: Unsupported model architecture

Not every model architecture is supported by every version of llama.cpp.

Check the official repository:

https://github.com/ggml-org/llama.cpp

Update to a version that supports the architecture.

Do not assume that renaming or converting an unsupported model will make it compatible.

### Problem: Quantization command fails

Example:

```text
llama-quantize: command not found
```

**Possible causes:**

- llama.cpp was not built.
- Executable is in another directory.
- The wrong build configuration was used.
- Executable name or path differs.

On Windows, locate the executable inside the build directory.

For example:

```powershell
Get-ChildItem .\build -Recurse -Filter llama-quantize.exe
```

Use the actual path returned by the command.

### Problem: Converted GGUF model produces poor output

**Possible causes:**

- Incorrect conversion settings.
- Unsupported model architecture.
- Incompatible tokenizer metadata.
- Incorrect chat template.
- Excessive quantization.
- Wrong prompt format.

**Solutions:**

- Follow the model-specific conversion instructions.
- Check the original model card.
- Verify the chat template.
- Compare against the original model or a higher-precision version.
- Try a different quantization variant.

## 8. Python Environment Troubleshooting

Python-based local LLM tools frequently depend on specific package versions.

### Problem: Python command is not recognized

Check:

```powershell
python --version
```

If unavailable, install Python and ensure it is accessible through PATH.

Official website:

https://www.python.org/downloads/

### Problem: Module not found

Example:

```text
ModuleNotFoundError: No module named 'transformers'
```

Install the missing package:

```powershell
python -m pip install transformers
```

For other packages, replace `transformers` with the required package name.

### Problem: Wrong Python environment

Check the Python executable:

```powershell
python -c "import sys; print(sys.executable)"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, check your execution policy before changing it. You can also use the virtual environment's Python executable directly without activating it.

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install project requirements:

```powershell
python -m pip install -r requirements.txt
```

### Problem: Package dependency conflicts

Example:

```text
ResolutionImpossible
```

**Solutions:**

1. Use a clean virtual environment.
2. Review package version requirements.
3. Avoid installing unrelated packages into the same environment.
4. Follow the official installation instructions for the tool.
5. Record working package versions when developing reproducible workflows.

Avoid randomly upgrading every dependency. That can introduce additional compatibility problems.

## 9. Model Download and Storage Problems

### Problem: Not enough disk space

Model files can be several gigabytes or larger.

Check available storage:

```powershell
Get-PSDrive -PSProvider FileSystem
```

On Linux:

```bash
df -h
```

Solutions:

- Remove unused model files.
- Use a drive with sufficient free space.
- Check where the application stores downloaded models.
- Keep enough extra space for model conversion and temporary files.

### Problem: Hugging Face download requires authentication

Some model repositories require authentication or acceptance of usage conditions.

Use the official Hugging Face CLI authentication workflow.

Documentation:

https://huggingface.co/docs/huggingface_hub/guides/cli

Never share access tokens or commit them to a Git repository.

### Problem: Git LFS files are incomplete

Some model repositories use Git Large File Storage.

Check that Git LFS is installed:

```powershell
git lfs version
```

Initialize it:

```powershell
git lfs install
```

Download LFS files:

```powershell
git lfs pull
```

Follow the repository's download instructions if it uses a different method.

## 10. Network and Port Troubleshooting

Local inference applications often expose HTTP APIs.

For example, Ollama normally uses port 11434.

Check whether the port is listening:

```powershell
Get-NetTCPConnection -LocalPort 11434 -State Listen
```

Check a local endpoint:

```powershell
Test-NetConnection localhost -Port 11434
```

If the connection fails:

1. Check whether the inference server is running.
2. Confirm the correct port.
3. Check whether another application is using that port.
4. Review firewall rules.
5. Verify the application's configured host address.

Do not disable your firewall as a general troubleshooting shortcut.

Avoid exposing local model APIs to public networks without appropriate access controls.

## 11. Useful Diagnostic Commands

### Windows PowerShell

| Purpose | Command |
|---|---|
| Python version | `python --version` |
| Pip version | `python -m pip --version` |
| GPU status | `nvidia-smi` |
| System memory | `Get-CimInstance Win32_OperatingSystem` |
| Ollama version | `ollama --version` |
| Ollama models | `ollama list` |
| Running Ollama models | `ollama ps` |
| Check API | `Invoke-RestMethod http://localhost:11434/api/tags` |
| Check port | `Test-NetConnection localhost -Port 11434` |
| Git status | `git status` |

### Linux

| Purpose | Command |
|---|---|
| System information | `uname -a` |
| Memory | `free -h` |
| CPU | `lscpu` |
| Disk space | `df -h` |
| GPU status | `nvidia-smi` |
| Ollama service | `systemctl status ollama` |
| Ollama models | `ollama list` |
| Running models | `ollama ps` |
| Check API | `curl http://localhost:11434/api/tags` |

## 12. How to Report a Useful Bug

If you cannot solve a problem, prepare a useful bug report.

Include:

- Operating system and version.
- CPU model.
- GPU model and VRAM, if applicable.
- System RAM.
- Inference software and version.
- Model name and quantization.
- Exact command or steps that caused the issue.
- Complete error message.
- What you already tried.

Example:

```text
## Environment

OS: Windows 11
CPU: [Your CPU]
GPU: [Your GPU]
RAM: [Your RAM]

## Software

Inference engine: Ollama
Version: [Version]
Model: [Model name and tag]

## Problem

The model fails to load.

## Error

[Paste the complete error message]

## Steps tried

1. Restarted Ollama.
2. Checked available memory.
3. Retried model loading.

## Expected result

The model should load and respond to prompts.
```

Before sharing logs, remove API keys, access tokens, private file paths, personal information, and other sensitive data.

## 13. When to Ask for Help

If a problem remains unresolved:

1. Search the official documentation.
2. Check the project's GitHub issues.
3. Search for the exact error message.
4. Check whether the issue is specific to your operating system or hardware.
5. Create a reproducible bug report.

Useful support resources:

- [Ollama GitHub Issues](https://github.com/ollama/ollama/issues)
- [LM Studio Documentation](https://lmstudio.ai/docs)
- [llama.cpp GitHub Issues](https://github.com/ggml-org/llama.cpp/issues)
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [PyTorch Forums](https://discuss.pytorch.org/)

---

## Further Reading

- [Getting Started with Local LLMs](getting-started.md)
- [Hardware Guide](hardware-guide.md)
- [Model Selection Guide](model-selection.md)
- [Quantization Guide](quantization.md)
- [Repository Structure](../STRUCTURE.md)

**Remember:** Troubleshoot systematically. Identify the failing component, collect diagnostic information, change one variable at a time,