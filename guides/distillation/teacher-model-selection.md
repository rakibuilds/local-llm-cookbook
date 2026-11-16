# Teacher Model Selection

Choosing the right teacher model is one of the most important decisions in a knowledge distillation project.

The teacher provides the knowledge, responses, or training signals used to train the student. A strong teacher can improve the quality of the training data, but model size alone does not determine whether a teacher is suitable.

This guide explains how to select teacher and student models for practical local LLM distillation projects.

## 1. Understand the Teacher and Student Roles

| Component | Responsibility |
|---|---|
| Teacher model | Generates responses or provides training signals. |
| Student model | Learns from the teacher and performs the target tasks. |
| Distillation dataset | Stores the examples or training signals used during learning. |

A teacher does not necessarily need to be dramatically larger than the student. The important consideration is whether it provides useful knowledge that the student can learn.

## 2. Define Your Distillation Goal

Before selecting models, define what the student should accomplish.

Examples:

| Goal | What to look for in a teacher |
|---|---|
| General instruction following | Strong instruction-following performance |
| Coding assistance | Reliable code generation and debugging |
| Question answering | Accurate, grounded responses |
| Summarization | Consistent and concise summaries |
| Assamese language tasks | Strong Assamese language understanding and generation |
| Domain-specific assistant | Relevant domain knowledge and reliable explanations |

Avoid selecting a model only because it is popular or has a large parameter count.

## 3. Choosing a Teacher Model

Evaluate potential teachers using the following criteria.

### 3.1 Task Quality

The teacher should perform well on the tasks you want the student to learn.

Test it with representative prompts before generating a large dataset.

Consider:

- Accuracy.
- Instruction following.
- Reasoning quality.
- Response consistency.
- Language coverage.
- Formatting reliability.

A teacher that performs poorly on the target task is unlikely to produce a high-quality distillation dataset.

### 3.2 Model Access

The available interface determines which distillation methods you can use.

| Teacher access | Available training signals |
|---|---|
| Text-generation API | Generated responses, if permitted |
| Local model through Ollama | Generated responses |
| Hugging Face model with accessible weights | Potentially logits and hidden states |
| Specialized training interface | Depends on exposed outputs and framework support |

**Important:** An API that returns only generated text supports response-based distillation, not necessarily logit-based distillation.

Check API terms, rate limits, data handling policies, and model licenses before using generated outputs for training.

### 3.3 License and Usage Restrictions

Always inspect the model's license before using it as a teacher.

Check whether the license permits:

- Generating training data.
- Using outputs to train other models.
- Commercial use, if relevant.
- Redistribution of generated datasets.
- Redistribution of the resulting student model.

Some model licenses contain restrictions that go beyond ordinary open-source software licenses.

Do not assume that publicly downloadable model weights are unrestricted.

### 3.4 Hardware Requirements

Teacher inference can require substantial memory, particularly for large models and long context lengths.

Consider:

- Model parameter count.
- Weight precision or quantization.
- Context length.
- KV cache requirements.
- Runtime overhead.
- GPU VRAM and system RAM.

A quantized teacher may be practical on limited hardware, but quantization can affect output quality.

For a response-based workflow, you can also generate teacher responses on another machine or through a permitted API and train the student locally.

## 4. Choosing a Student Model

The student should match your deployment requirements.

Consider:

1. Model architecture.
2. Parameter count.
3. Context length.
4. Tokenizer compatibility.
5. Supported training framework.
6. Hardware requirements.
7. Model license.
8. Expected inference performance.

A smaller model is not automatically a suitable student. It must have sufficient capacity to learn the target behavior.

### Student selection considerations

| Student characteristic | Why it matters |
|---|---|
| Smaller parameter count | Can reduce weight memory requirements |
| Compatible architecture | Simplifies training and implementation |
| Suitable tokenizer | Important for training and teacher/student token-level alignment |
| Supported training framework | Makes experimentation easier |
| Appropriate context length | Supports the intended application |
| Permissive license | Helps with intended distribution and usage |

For response-based distillation, teacher and student tokenizers do not necessarily need to match because the student can learn from text examples using its own tokenizer.

For logit-based distillation, token-level alignment and compatible output spaces become important.

## 5. Teacher and Student Model Size

A larger teacher can provide stronger training signals, but there is no universal teacher-to-student size ratio.

For example, a project might explore:

| Teacher | Student | Experiment |
|---|---|---|
| 7B model | 1.5B model | General instruction distillation |
| 14B model | 3B model | Domain-specific assistant |
| 3B model | 0.5B model | Lightweight task-focused model |

These are illustrative combinations, not guaranteed performance recommendations. Check architecture compatibility, licenses, and hardware before choosing actual model versions.

### A practical selection principle

Choose:

- A teacher that performs reliably on your target tasks.
- A student that can run within your deployment budget.
- A training dataset that represents the actual tasks the student will encounter.

Evaluate the student rather than assuming that a particular size ratio guarantees success.

## 6. Discovering Models on Hugging Face

[Hugging Face Model Hub](https://huggingface.co/models) provides a searchable collection of models.

Useful filters include:

- Model architecture.
- Parameter count.
- Task.
- Library compatibility.
- License.
- Downloads and community activity.

Before downloading a model, inspect its model card.

Check:

- Intended use.
- Supported languages.
- Training details.
- Known limitations.
- License.
- Available checkpoints.
- Recommended inference and training methods.

A high download count does not guarantee suitability for your task.

## 7. Running a Local Teacher with Ollama

[Ollama](https://ollama.com/) provides a convenient way to run supported language models locally.

For example, you can explore a small Qwen model:

```powershell
ollama run qwen3:0.6b
```

This is a small-model experimentation example, not a recommendation that it will be a strong teacher for every task.

Useful commands:

```powershell
ollama list
ollama show qwen3:0.6b
ollama ps
```

To generate teacher responses through the local Ollama API, see the [Ollama guide](../running-local-llms/ollama.md).

Remember that ordinary generated text is suitable for response-based distillation. Access to internal logits or hidden states requires additional model and runtime support.

## 8. Evaluate Teachers Before Dataset Generation

Do not generate thousands of examples before checking teacher quality.

Create a small evaluation set containing representative prompts.

For example:

```json
[
  {
    "task": "explanation",
    "prompt": "Explain model quantization in simple terms."
  },
  {
    "task": "summarization",
    "prompt": "Summarize the purpose of local language models."
  },
  {
    "task": "coding",
    "prompt": "Write a Python function that reverses a string."
  }
]
```

Evaluate the teacher's responses for:

- Correctness.
- Relevance.
- Clarity.
- Completeness.
- Instruction compliance.
- Consistency.

For factual tasks, use independently verified answers or references wherever possible.

## 9. Teacher Selection Checklist

Before committing to a teacher/student combination, review this checklist:

- [ ] The target task is clearly defined.
- [ ] The teacher performs well on representative prompts.
- [ ] The teacher's access method supports the intended distillation technique.
- [ ] Model licensing and usage restrictions have been checked.
- [ ] The student architecture is supported by the training framework.
- [ ] The student has enough capacity for the target task.
- [ ] Hardware requirements are realistic.
- [ ] Tokenizer compatibility has been considered where necessary.
- [ ] A small evaluation dataset is available.
- [ ] The teacher and student can be compared consistently.

## 10. Recommended Starting Strategy

For beginners, start with response-based distillation.

A practical workflow is:

1. Select a teacher that performs well on a narrow task.
2. Select a smaller student model.
3. Prepare a small collection of representative prompts.
4. Generate teacher responses.
5. Review and clean the generated examples.
6. Fine-tune the student on the prepared dataset.
7. Evaluate the student against the teacher and independent references.
8. Optimize the student for deployment if needed.

This approach is easier to understand and implement than logit-based or feature-based distillation.

## 11. Next Steps

Continue with:

- [Knowledge Distillation](knowledge-distillation.md)
- [Dataset Generation](dataset-generation.md)
- [Student Model Training](student-model-training.md)
- [Distillation Examples](../../examples/distillation/README.md)

Related guides:

- [Model Selection](../../docs/model-selection.md)
- [Hardware Guide](../../docs/hardware-guide.md)
- [Quantization](../../docs/quantization.md)

---

**Part of [Local LLM Cookbook](../../README.md)** — a practical, hardware-aware open-source resource for running, fine-tuning, quantizing, and deploying language models locally.