# Local LLM Glossary

This glossary explains common terms used in Large Language Models (LLMs), local AI development, model inference, fine-tuning, and deployment.

It is designed for beginners and developers who want to understand the terminology used throughout this cookbook.

Terms are organized alphabetically for easy reference.

## A

### Adapter

A small set of additional trainable parameters used to modify an existing model without updating all its original weights.

Adapters are commonly used with LoRA and other parameter-efficient fine-tuning techniques.

### AI (Artificial Intelligence)

A field of computer science focused on building systems that perform tasks associated with human intelligence, such as reasoning, learning, language understanding, and problem-solving.

### API (Application Programming Interface)

A defined interface that allows software applications to communicate with one another.

Local inference engines such as Ollama can expose APIs that applications use to send prompts and receive model responses.

### Autoregressive Model

A model that generates output sequentially by predicting the next token based on previously available tokens.

Many popular language models use autoregressive generation.

## B

### Batch Size

The number of inputs processed together during a model operation.

Larger batch sizes can improve throughput but generally require more memory.

### BF16 (Brain Floating Point 16)

A 16-bit floating-point numerical format commonly used in machine learning.

It has a similar memory footprint to FP16 but a wider exponent range.

### Benchmark

A standardized test or evaluation used to measure model performance.

Benchmarks may evaluate coding, reasoning, language understanding, mathematics, or other capabilities.

Benchmark results do not necessarily predict performance on every real-world task.

### Base Model

A model trained primarily to predict the next token, before being adapted for instruction-following or conversational use.

Base models are commonly used as starting points for further training and fine-tuning.

## C

### Checkpoint

A saved representation of a model's parameters and associated training state or metadata.

Checkpoints are used to resume training, distribute models, and preserve particular stages of development.

### CUDA

A parallel computing platform and programming model developed by NVIDIA.

CUDA enables supported software to perform computations on compatible NVIDIA GPUs.

### Context Length

The maximum number of tokens a model can process within its configured context window.

Context length includes the relevant input and generated conversation history.

A longer context can support larger documents and conversations but may increase memory consumption.

### CPU (Central Processing Unit)

The main general-purpose processor in a computer.

A CPU can run local LLM inference, although performance depends on processor architecture, memory bandwidth, model size, and inference software.

### CPU Inference

Running model computations using the computer's CPU instead of relying entirely on a GPU.

CPU inference is useful on systems without a supported dedicated GPU.

## D

### Dataset

A collection of data used for training, fine-tuning, validation, or evaluation.

Datasets may contain text, conversations, instructions, code, images, audio, or other information.

### Decoding

The process of selecting tokens from a model's output distribution to generate a response.

Common decoding approaches include greedy decoding, sampling, and beam search.

### Deployment

Making a trained or configured model available for use by an application, service, or other users.

Deployment may involve a local machine, private server, or hosted infrastructure.

### Distillation

A technique in which a smaller student model learns from the outputs or behavior of a larger teacher model.

It can help create smaller models that approximate some capabilities of larger models.

## E

### Embedding

A numerical representation of text or other data in a vector space.

Embeddings help applications compare semantic similarity, retrieve relevant documents, and build search systems.

### Epoch

One complete pass through a training dataset.

Fine-tuning workflows often specify a number of epochs to control how many times the model processes the training data.

### Evaluation

The process of measuring how well a model performs on selected tasks or datasets.

Evaluation can include accuracy, response quality, latency, throughput, and other metrics.

### ExLlama

A family of inference implementations and tools designed for efficient execution of supported transformer models, particularly quantized models on compatible NVIDIA GPUs.

Check the current project documentation for supported formats and architectures.

## F

### Fine-Tuning

The process of adapting an existing pretrained model using additional training data.

Fine-tuning can help a model specialize in a particular domain, task, style, or instruction format.

### FP16 (Floating Point 16)

A 16-bit floating-point format used in machine learning.

It generally requires half the weight storage of FP32.

### FP32 (Floating Point 32)

A 32-bit floating-point format commonly used for numerical computation and some model training workflows.

It requires more memory than FP16 or BF16.

### Framework

A collection of software libraries and tools that simplify building, training, or running machine learning models.

Examples include PyTorch and TensorFlow.

## G

### GGML

An open-source tensor library and historical project name associated with the development of llama.cpp.

GGML is also commonly encountered in older model-format discussions.

Do not confuse the GGML library with GGUF, the newer model file format commonly used by llama.cpp.

### GGUF

GPT-Generated Unified Format.

A file format used by llama.cpp and compatible applications to store model weights and metadata.

GGUF supports multiple tensor types and quantization formats.

### GPU (Graphics Processing Unit)

A processor designed for parallel computation.

GPUs are widely used for machine learning because many model operations can be executed efficiently in parallel.

### GPU Offloading

Moving some or all model computations or layers onto a GPU.

Offloading can improve inference performance, but the result depends on GPU memory, hardware support, and runtime implementation.

## H

### Hugging Face

A platform and ecosystem for sharing and working with machine learning models, datasets, and related tools.

Website: https://huggingface.co/

### Hyperparameter

A configuration value that controls a training or inference process.

Examples include learning rate, batch size, number of epochs, temperature, and context length.

## I

### Inference

The process of using a trained model to generate predictions or responses.

For an LLM, inference usually means providing input text and generating output tokens.

Inference does not normally update the model's learned weights.

### Instruction-Tuned Model

A model that has been adapted to follow user instructions.

Instruction-tuned models are generally more suitable for conversational assistants and interactive applications than base models.

### INT4

A four-bit integer representation used in some model quantization methods.

It can significantly reduce weight storage, although quality and performance depend on the quantization method and runtime.

### INT8

An eight-bit integer representation used in quantization and other machine learning computations.

It generally requires less memory than FP16 weight storage.

## J

### JSON (JavaScript Object Notation)

A structured text format commonly used to exchange data between applications.

JSON is frequently used for training datasets, configuration files, and model APIs.

Example:

```json
{
  "role": "user",
  "content": "Explain local LLM inference."
}
```

## K

### KV Cache (Key-Value Cache)

A memory mechanism used by transformer models during autoregressive inference.

It stores previously calculated attention keys and values so the model does not need to recompute them for every new token.

KV cache memory requirements can increase significantly with context length, model architecture, and batch size.

### Knowledge Distillation

A training approach where a smaller model learns from a larger model's predictions or generated outputs.

It is often used to create smaller models with useful capabilities inherited from larger systems.

## L

### Learning Rate

A hyperparameter that controls the size of parameter updates during training.

An unsuitable learning rate can lead to unstable training or poor convergence.

### llama.cpp

An open-source project that enables LLM inference across different hardware platforms, including CPUs and supported GPUs.

It supports GGUF models and several quantization methods.

Repository: https://github.com/ggml-org/llama.cpp

### LM Studio

A desktop application for discovering, downloading, and running compatible local language models.

Website: https://lmstudio.ai/

### LoRA (Low-Rank Adaptation)

A parameter-efficient fine-tuning technique that introduces small trainable low-rank matrices into selected model layers.

LoRA allows adaptation without updating all original model parameters.

### Local LLM

A language model executed on hardware controlled by the user, such as a personal computer or private server.

Local inference can reduce dependence on external inference APIs, although model downloads and optional integrations may still require internet access.

### Loss Function

A mathematical function that measures the difference between a model's predictions and the expected targets during training.

Training generally aims to minimize the loss.

## M

### Machine Learning

A branch of artificial intelligence in which systems learn patterns from data.

Deep learning is a subset of machine learning based on neural networks with multiple layers.

### Model Architecture

The structural design of a neural network.

For LLMs, architecture determines how components such as attention layers, feed-forward networks, and token embeddings are organized.

### Model Card

Documentation describing a machine learning model.

A model card may include its intended uses, limitations, architecture, training information, evaluation results, and license.

### Model Parameters

Learned numerical values inside a model.

Parameter count is commonly used to describe model size, such as 1.5B, 7B, or 70B.

A larger parameter count does not automatically guarantee better performance.

### Model Weights

The learned numerical values that determine a model's behavior.

Model weights are stored in checkpoint files or other model formats.

### Multimodal Model

A model designed to process or generate more than one type of data, such as text, images, audio, or video.

Supported modalities vary between models.

## N

### Neural Network

A computational model composed of interconnected mathematical units, commonly organized into layers.

Neural networks are the foundation of modern deep learning.

### Next-Token Prediction

The process of predicting the next token based on the preceding sequence.

It is a central training objective for many autoregressive language models.

## O

### Ollama

A tool for downloading, running, and managing language models locally.

It also provides an API that applications can use to interact with supported models.

Website: https://ollama.com/

### Open Weights

A term describing models whose learned weights are publicly available.

Open weights do not necessarily mean that training data, training code, or every aspect of the model development process is open.

Always check the model's license and usage terms.

## P

### Parameter-Efficient Fine-Tuning (PEFT)

A family of techniques that adapt pretrained models by training a relatively small number of additional parameters instead of updating every model weight.

LoRA and QLoRA are examples.

### Parameters

See [Model Parameters](#model-parameters).

### Perplexity

A metric commonly used to evaluate how well a language model predicts a sequence of text.

Lower perplexity generally indicates better predictive performance on the evaluated dataset, but it does not guarantee better real-world task performance.

### Pretrained Model

A model that has already undergone an initial training process on a large dataset.

It can serve as the starting point for inference, instruction tuning, or further fine-tuning.

### PyTorch

An open-source machine learning framework widely used for developing, training, and running neural networks.

Website: https://pytorch.org/

## Q

### QLoRA (Quantized Low-Rank Adaptation)

A parameter-efficient fine-tuning approach that combines quantized model weights with trainable LoRA adapters.

It can reduce memory requirements compared with full-precision fine-tuning.

QLoRA does not eliminate the need for sufficient memory, compatible hardware, and suitable training software.

### Quantization

The process of representing numerical values with lower precision to reduce storage and memory requirements.

LLM quantization commonly reduces the precision of model weights.

### Q4_K_M

A commonly used four-bit GGUF quantization variant from the llama.cpp K-quant family.

It is often selected as a practical balance between model size and quality.

### Q8_0

An eight-bit GGUF quantization type.

It generally uses more storage than four-bit quantization while retaining relatively high weight precision.

## R

### RAM (Random Access Memory)

System memory used by the operating system and running applications.

Local LLM inference may use RAM to store model weights, runtime data, and other information.

### RAG (Retrieval-Augmented Generation)

An architecture that retrieves relevant external information and provides it to a language model as additional context.

RAG is commonly used to build document question-answering systems and knowledge assistants.

### Repository

A location where project files and version history are maintained.

Git repositories are commonly used to manage source code, documentation, and collaboration.

### Runtime

The software environment responsible for executing a model.

Examples include Ollama, llama.cpp, and compatible inference engines.

## S

### Sampling

A process for selecting generated tokens from a model's probability distribution.

Sampling settings can influence the diversity and predictability of generated text.

### SLM (Small Language Model)

A language model designed with a relatively small parameter count compared with larger language models.

SLMs can be useful for local deployment and hardware-constrained applications.

There is no universally accepted parameter threshold separating SLMs from LLMs.

### System Prompt

An instruction or message that establishes the behavior, role, or constraints of an AI assistant.

System prompts are commonly used in conversational applications.

## T

### Temperature

An inference parameter that controls the randomness of token sampling.

- Lower values generally produce more predictable outputs.
- Higher values generally allow more varied outputs.

Temperature does not improve the underlying knowledge or reasoning ability of a model.

### Token

A unit of text processed by a language model.

Tokens can represent complete words, parts of words, punctuation, or other text fragments.

The number of characters in a word does not directly determine its token count.

### Tokenizer

A component that converts text into tokens and converts generated tokens back into text.

Different models can use different tokenization systems.

### Tokens Per Second (TPS)

A performance measurement describing how many tokens an inference system generates per second.

TPS is commonly used to compare local inference performance, but comparisons should account for hardware, model, context, and generation settings.

### Transformer

A neural network architecture based on attention mechanisms.

Transformers are the foundation of many modern language models.

### Training

The process of adjusting model parameters using data and an optimization algorithm.

Training can involve pretraining a model from scratch or fine-tuning an existing model.

## U

### Unsloth

A software project providing tools and optimizations for efficient fine-tuning of supported language models.

It is commonly used for LoRA and QLoRA workflows.

Repository: https://github.com/unslothai/unsloth

## V

### VRAM (Video Random Access Memory)

Dedicated memory available on a GPU.

VRAM stores model weights and other data needed for GPU-based inference.

The amount of available VRAM is an important consideration when selecting local models.

### Vector Database

A database designed to store and retrieve vector representations.

Vector databases are commonly used in semantic search and RAG applications.

## W

### Weights

See [Model Weights](#model-weights).

### Weight Quantization

A quantization technique that reduces the precision used to represent model weights.

It is one of the most common approaches to reducing local LLM memory requirements.

## X

### Xformers

A collection of optimized transformer building blocks developed to improve performance and memory efficiency for supported workloads.

Its compatibility depends on the software stack, hardware, and model implementation.

## Z

### Zero-Shot Learning

The ability of a model to perform a task without being explicitly fine-tuned on examples of that particular task.

Zero-shot prompting commonly relies on instructions provided directly in the input.

---

## Related Guides

- [Getting Started with Local LLMs](getting-started.md)
- [Hardware Guide](hardware-guide.md)
- [Model Selection Guide](model-selection.md)
- [Quantization Guide](quantization.md)
- [Troubleshooting Guide](troubleshooting.md)
- [Repository Structure](../STRUCTURE.md)

## Contributing to This Glossary

This glossary is intended to grow with the local LLM ecosystem.

Contributions are welcome. When adding a new term:

1. Place it under the appropriate alphabetical heading.
2. Explain it in beginner-friendly language.
3. Avoid unnecessary jargon.
4. Include an example or related guide when useful.
5. Verify technical accuracy before submitting changes.

**Goal:** Make local LLM development easier to understand, explore, and practice for everyone.