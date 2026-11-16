# Getting Started with Local LLMs

Welcome to Local LLM Cookbook!

This guide will help you understand local Large Language Models (LLMs), prepare your computer, install the required tools, and run your first AI model locally.

No cloud inference API or paid AI subscription is required.

## 1. What Is a Local LLM?

A Large Language Model (LLM) is an AI model trained to understand and generate human language.

A local LLM runs directly on your own computer instead of sending prompts to a remote AI service.

Running models locally gives you:

- **Privacy:** Your prompts can remain on your own machine.
- **Control:** Choose and configure the models you want.
- **Offline access:** Use downloaded models without an internet connection.
- **Customization:** Experiment with quantization and fine-tuning.
- **Learning:** Understand how modern AI inference works.

Local inference still requires sufficient hardware resources, and downloaded models may have their own usage licenses.

## 2. What Do You Need?

You do not need an expensive workstation to get started.

Your hardware requirements depend on the model size, quantization, context length, and inference method.

### Basic requirements

| Component | Recommendation |
|---|---|
| Operating system | Windows, Linux, or macOS |
| RAM | 8 GB minimum for small-model experimentation; 16 GB or more recommended |
| Storage | SSD with sufficient free space for models |
| GPU | Optional for CPU inference; helpful for faster inference |
| Internet | Required to download tools and models |

These are starting recommendations, not guarantees that every model will run.

### Understanding RAM and VRAM

- **RAM:** System memory used by your operating system and applications.
- **VRAM:** Dedicated GPU memory used for model weights and other GPU workloads.

A model can run without a dedicated GPU using system RAM and CPU inference, although performance may be slower.

## 3. Choose Your Local LLM Tool

Several tools make running local models easier.

| Tool | Best for |
|---|---|
| Ollama | Running models through a simple CLI and API |
| LM Studio | Desktop interface for discovering and chatting with models |
| llama.cpp | Efficient GGUF model inference |
| Hugging Face | Discovering and working with open models |

For this beginner tutorial, we will use Ollama.

## 4. Install Ollama

Ollama makes it easy to download and run supported language models locally.

Official website:

https://ollama.com/

### Windows

1. Visit the Ollama website.
2. Download the Windows installer.
3. Run the installer.
4. Open PowerShell.
5. Verify the installation:

```powershell
ollama --version
```

If the command is recognized, Ollama is installed.

### Linux

Download and install Ollama using its official installation instructions:

https://ollama.com/download/linux

After installation, verify:

```bash
ollama --version
```

### macOS

Download Ollama for macOS:

https://ollama.com/download/mac

Install the application and verify the CLI installation:

```bash
ollama --version
```

## 5. Download Your First Model

For a lightweight starting point, try a small model such as Qwen3 0.6B.

Run:

```bash
ollama run qwen3:0.6b
```

Ollama will download the model if it is not already available locally.

Model availability and hardware performance can vary. Check the model's requirements before downloading larger models.

## 6. Run Your First Prompt

Once the model starts, enter:

```text
Explain what a Large Language Model is in simple terms.
```

The model should generate a response directly on your computer.

You can continue the conversation by entering additional prompts.

To exit the interactive session, type:

```text
/bye
```

## 7. Explore Your Downloaded Models

List locally available models:

```bash
ollama list
```

Show currently running models:

```bash
ollama ps
```

Remove a model you no longer need:

```bash
ollama rm qwen3:0.6b
```

## 8. Use Ollama Through Its Local API

Ollama also provides a local HTTP API, allowing applications to communicate with downloaded models.

By default, its local API is available at:

```text
http://localhost:11434
```

Example using Python:

Install the requests library:

```bash
pip install requests
```

Create a Python file named `chat.py`:

```python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen3:0.6b",
        "messages": [
            {
                "role": "user",
                "content": "Explain local AI inference."
            }
        ],
        "stream": False
    },
    timeout=120
)

response.raise_for_status()

result = response.json()
print(result["message"]["content"])
```

Run it:

```bash
python chat.py
```

Make sure Ollama is running and the selected model is downloaded before executing the script.

## 9. Common Problems

### Ollama command not found

- Verify that Ollama is installed.
- Restart your terminal.
- Check that Ollama is available in your system PATH.

### Model runs very slowly

- Try a smaller model.
- Use a more aggressively quantized model where appropriate.
- Close unnecessary applications.
- Check CPU, RAM, and GPU utilization.

### Out of memory

- Stop other memory-intensive applications.
- Choose a smaller model.
- Reduce the context length where supported.
- Consider CPU inference if GPU memory is insufficient.

### Model download fails

- Check your internet connection.
- Verify the model name.
- Retry the download.

## 10. What's Next?

Now that you have run your first local model, explore the next topics:

- [Hardware Guide](hardware-guide.md)
- [Model Selection](model-selection.md)
- [Quantization](quantization.md)
- [Troubleshooting](troubleshooting.md)

Continue learning through the practical tutorials in the `guides/` directory.

---

**Local LLM Cookbook** — Learn, experiment, and build with AI models running on your own hardware.