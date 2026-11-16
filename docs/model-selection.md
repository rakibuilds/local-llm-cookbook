# Choosing the Right Local LLM

Choosing a local Large Language Model (LLM) is one of the most important steps when building a local AI setup.

The best model is not necessarily the largest or most popular one. It is the model that provides the right balance of quality, speed, memory usage, compatibility, and licensing for your particular needs.

This guide explains how to compare models and choose one that fits your hardware and use case.

## 1. Understand Model Sizes

LLMs are commonly described by their number of parameters.

A parameter is a value learned during training. Larger parameter counts generally provide greater capacity, but they also increase memory requirements and computational cost.

| Model size | General category |
|---|---|
| 0.5B–1B | Very small |
| 1B–3B | Small |
| 3B–8B | Medium |
| 8B–14B | Larger |
| 14B–32B | Large |
| 32B–70B | Very large |
| 70B+ | Extremely large |

These categories are approximate. Architecture, training quality, quantization, context length, and inference software all affect actual performance.

### Smaller models

Smaller models are useful for:

- Basic conversational assistants.
- Simple text generation.
- Lightweight coding assistance.
- Classification and extraction.
- Experimenting on limited hardware.

### Larger models

Larger models may offer better capabilities for:

- Complex reasoning.
- Advanced coding tasks.
- Detailed writing and analysis.
- More demanding instruction-following tasks.

However, parameter count alone does not determine model quality. A smaller, well-trained model can outperform a larger model on particular tasks.

## 2. Understand Model Families

Several model families are commonly used for local inference.

### Qwen

Qwen is a family of language and multimodal models developed by Alibaba.

Common use cases include:

- General-purpose chat.
- Coding assistance.
- Reasoning.
- Multilingual applications.
- Local AI experimentation.

Qwen models are available in different sizes and variants. Check the individual model card for supported languages, capabilities, hardware requirements, and licensing.

Official resources: https://huggingface.co/Qwen

### Llama

Llama is a family of models developed by Meta.

Common use cases include:

- General-purpose assistants.
- Text generation.
- Coding.
- Research and experimentation.
- Building applications with local inference.

Different Llama generations and variants have different capabilities and licensing conditions.

Official resources: https://www.llama.com/

### Gemma

Gemma is a family of open-weight models developed by Google.

Common use cases include:

- Lightweight AI applications.
- General-purpose text generation.
- Instruction-following.
- Research and experimentation.

Check the specific release for its supported features, hardware requirements, and usage terms.

Official resources: https://ai.google.dev/gemma

### Mistral

Mistral AI publishes several model families, including compact and larger language models.

Common use cases include:

- General-purpose assistants.
- Coding.
- Text generation.
- Local AI applications.

Capabilities and licensing vary between releases.

Official resources: https://mistral.ai/

**Important:** Model availability, architecture, supported languages, and licensing change between releases. Always read the individual model card and license before using a model in a project.

## 3. Base Models vs Instruction-Tuned Models

Not every language model is designed for direct conversation.

### Base models

A base model is trained primarily to predict the next token in a sequence.

They are useful for:

- Further fine-tuning.
- Research.
- Custom training workflows.
- Specialized language-model development.

A base model may not reliably follow conversational instructions without additional training or prompting.

### Instruction-tuned models

Instruction-tuned models are adapted to follow user instructions and respond in a conversational format.

They are generally better suited for:

- Chatbots.
- Personal assistants.
- Coding assistants.
- Question answering.
- Everyday local AI applications.

**Recommendation:** If you are starting with Ollama or LM Studio, choose an instruction-tuned model unless you have a specific reason to use a base model.

## 4. Understand Quantization

Quantization reduces the numerical precision used to store model weights.

This can significantly reduce memory requirements and make larger models practical on consumer hardware.

Common formats and precision levels include:

| Format | General description |
|---|---|
| FP32 | 32-bit floating point |
| FP16 | 16-bit floating point |
| BF16 | 16-bit brain floating point |
| INT8 | 8-bit integer quantization |
| INT4 | 4-bit integer quantization |
| GGUF | File format commonly used by llama.cpp-based inference tools |

GGUF is a model file format, not a quantization level. GGUF files can contain models using different quantization types.

### Common GGUF quantization variants

| Variant | General characteristics |
|---|---|
| Q2_K | Very small, greater quality reduction |
| Q3_K_M | Small, more quality reduction |
| Q4_K_M | Good size and quality balance |
| Q5_K_M | Higher quality, larger file |
| Q6_K | Higher precision, larger file |
| Q8_0 | Near-original quality in many cases, larger file |
| F16 | Half-precision weights, substantially larger |

These are common llama.cpp quantization labels. Exact formats and available variants depend on the model and conversion tools.

### Which quantization should you choose?

- Choose Q4_K_M when memory is limited and you need a practical balance.
- Choose Q5_K_M when you have additional memory and want to preserve more quality.
- Choose Q6_K or Q8_0 when memory is less restrictive.
- Consider lower-bit formats only when necessary, and evaluate their quality on your actual tasks.

Quantization can affect accuracy and output quality. The impact varies by model and task.

## 5. Estimate Model Memory Requirements

A model's parameter count gives a rough estimate of the memory required for its weights.

For example:

| Model size | FP16 weight estimate | 4-bit weight estimate |
|---|---:|---:|
| 0.5B | ~1 GB | ~0.25–0.4 GB |
| 1.5B | ~3 GB | ~0.75–1 GB |
| 3B | ~6 GB | ~1.5–2 GB |
| 7B | ~14 GB | ~3.5–4.5 GB |
| 8B | ~16 GB | ~4–5 GB |
| 14B | ~28 GB | ~7–9 GB |
| 32B | ~64 GB | ~16–20 GB |
| 70B | ~140 GB | ~35–45 GB |

These are approximate weight-storage estimates, not complete runtime requirements.

Actual memory usage also depends on:

- Model architecture.
- Quantization format.
- Context length.
- KV cache.
- Batch size.
- Runtime buffers.
- GPU offloading.
- Inference software.

Leave additional memory for the operating system and other applications.

For more detailed hardware guidance, see the [Hardware Guide](hardware-guide.md).

## 6. Choose a Model Based on Your Hardware

Use your available RAM and VRAM to narrow down the options.

| Hardware situation | Models worth exploring |
|---|---|
| 4 GB RAM | Very small models, if the operating system leaves enough memory |
| 8 GB RAM | Small quantized models |
| 16 GB RAM | Small and selected medium quantized models |
| 32 GB RAM | Medium models and some larger quantized models |
| 16 GB RAM + 8 GB VRAM | Small and selected medium models with GPU offloading |
| 32 GB RAM + 12 GB VRAM | Medium quantized models and partial GPU offloading |
| 64 GB RAM + 24 GB VRAM | Larger quantized models and more GPU offloading |

These are starting points, not guaranteed compatibility recommendations. Actual feasibility depends on the model, context size, runtime, and available memory.

A model can sometimes run using system RAM even when it does not fit entirely in VRAM, but performance may be significantly slower.

## 7. Choose a Model Based on Your Use Case

Different tasks benefit from different model capabilities.

| Use case | What to look for |
|---|---|
| General chat | Instruction following and conversational quality |
| Coding | Code generation, debugging, and programming-language support |
| Reasoning | Multi-step problem-solving capabilities |
| Writing | Language quality, coherence, and instruction following |
| Translation | Strong performance in both source and target languages |
| Multilingual applications | Good coverage of your required languages |
| Data extraction | Structured output and reliable instruction following |
| Local AI agents | Tool calling, structured output, and reliable instruction following |

For multilingual projects, evaluate the model using actual examples from your target languages.

Do not assume that a model performs equally well across every language.

## 8. Understand Context Length

Context length describes how much input and generated text a model can process within a single context window.

It is commonly measured in tokens.

For example:

- 4K context: approximately 4,096 tokens.
- 8K context: approximately 8,192 tokens.
- 32K context: approximately 32,768 tokens.
- 128K context: approximately 131,072 tokens.

Actual supported context lengths depend on the model and its configuration.

Longer context windows can be useful for:

- Large documents.
- Code repositories.
- Long conversations.
- Retrieval-augmented generation (RAG).
- Document analysis.

However, longer context generally increases memory consumption, especially through the KV cache.

Do not assume that a model's advertised maximum context length will be practical on your hardware.

## 9. Where to Find Local LLMs

Use reputable sources when downloading models.

### Hugging Face

Hugging Face hosts model repositories, model cards, quantized variants, and datasets.

Website: https://huggingface.co/models

Useful filters include:

- Model architecture.
- Parameter size.
- Task.
- Library.
- License.
- Quantization format.

Always verify that a repository is maintained by the original developer or a trusted publisher.

### Ollama Library

Ollama provides a model library with convenient local installation and execution.

Website: https://ollama.com/library

Example:

```powershell
ollama run qwen3:0.6b
```

The model name is an example. Check the current library for availability and hardware suitability.

### LM Studio

LM Studio provides a graphical interface for discovering and running compatible local models.

Website: https://lmstudio.ai/

It can be useful for users who prefer downloading and testing models through a desktop application.

## 10. Evaluate a Model Before Choosing It

Do not choose a model based only on benchmark scores or popularity.

Test it against your actual requirements.

Consider the following:

1. Does it understand your instructions?
2. Does it produce useful and accurate answers?
3. Does it support your required languages?
4. Does it perform well on your coding or reasoning tasks?
5. Does it fit within your available memory?
6. Is its response speed acceptable?
7. Does it support the inference software you want to use?
8. Does its license allow your intended use?

### Example evaluation prompts

General chat:

```text
Explain how a CPU and GPU work together when running a local language model.
```

Coding:

```text
Write a Python function that validates an email address.
Include a few test cases.
```

Reasoning:

```text
A shop sells a product for ₹800 after applying a 20% discount.
What was its original price? Explain the calculation.
```

Multilingual:

```text
Explain what a local language model is in Assamese.
Use simple language and give one practical example.
```

Use the same prompts when comparing multiple models. This makes your evaluation more consistent.

## 11. Understand Model Licensing

A model being publicly downloadable does not automatically mean it is unrestricted for every use.

Before using a model, check:

- Its license.
- Commercial-use permissions.
- Redistribution requirements.
- Attribution requirements.
- Acceptable-use policies.
- Additional restrictions.
- Any conditions imposed by the model publisher.

Also check the licenses of datasets, adapters, and other components used in your application.

**Important:** The license of this cookbook does not change the license of any third-party model or dataset.

## 12. Practical Model Selection Checklist

Before downloading a model, review these questions:

- [ ] Is the model appropriate for my use case?
- [ ] Is it instruction-tuned if I need conversational interaction?
- [ ] Does it fit within my available RAM and VRAM?
- [ ] Have I selected an appropriate quantization?
- [ ] Is its context length sufficient?
- [ ] Does it support my inference software?
- [ ] Does it perform well in my required language?
- [ ] Have I checked its model card and license?
- [ ] Have I tested its response quality and speed?

## 13. Recommended Learning Path

If you are new to local LLMs, follow this progression:

1. Start with a small instruction-tuned model.
2. Run it using Ollama or LM Studio.
3. Test its quality with your own prompts.
4. Compare a few quantization variants.
5. Monitor RAM, VRAM, and response speed.
6. Experiment with a larger model if your hardware allows.
7. Explore fine-tuning only after understanding inference.

## Further Reading

- [Getting Started with Local LLMs](getting-started.md)
- [Hardware Guide](hardware-guide.md)
- [Quantization Guide](quantization.md)
- [Troubleshooting](troubleshooting.md)
- [Repository Structure](../STRUCTURE.md)

---

**Remember:** The best local LLM is the one that performs your intended tasks reliably, runs comfortably on your hardware, and has licensing terms that support your intended use.