# Model Distillation Examples

This directory contains practical Python examples demonstrating response-based knowledge distillation for Large Language Models (LLMs).

The examples follow a simple workflow:

1. Generate responses using a teacher model.
2. Prepare a dataset for student training.
3. Fine-tune a smaller student model.
4. Evaluate the trained student against the original model.

The initial implementation uses local-first tools and does not require a cloud inference API.

## 1. Project Structure

```text
examples/
└── distillation/
    ├── README.md
    ├── generate-teacher-data.py
    ├── train-student.py
    └── evaluate-student.py
```

Generated datasets and training outputs should be stored separately from the source scripts.

Recommended working layout:

```text
examples/
└── distillation/
    ├── data/
    │   ├── prompts.jsonl
    │   ├── train.jsonl
    │   ├── validation.jsonl
    │   └── test.jsonl
    ├── outputs/
    │   └── student-distilled/
    ├── generate-teacher-data.py
    ├── train-student.py
    ├── evaluate-student.py
    └── README.md
```

Generated data and model checkpoints should not be committed unless they are intentionally small, appropriately licensed, and useful as reproducible examples.

## 2. Requirements

### Software

- Python 3.10 or later.
- Ollama for local teacher inference.
- PyTorch.
- Hugging Face Transformers.
- Hugging Face Datasets.
- TRL.
- PEFT.
- Accelerate.
- Requests.

Training dependencies:

```powershell
python -m pip install torch transformers datasets trl peft accelerate
```

Teacher-generation dependency:

```powershell
python -m pip install requests
```

Install a PyTorch build compatible with your operating system and hardware. For GPU training, follow the official PyTorch installation instructions:

https://pytorch.org/get-started/locally/

### Hardware

Hardware requirements depend on the selected teacher, student, sequence length, batch size, precision, and training strategy.

- Teacher generation can run on CPU or GPU, depending on model size.
- Student training generally benefits from a supported GPU.
- LoRA can reduce training memory requirements.
- QLoRA requires compatible quantization libraries and hardware.

Do not assume that every GPU supports every training configuration. Windows support for some quantization libraries differs from Linux.

## 3. Step One: Start a Local Teacher Model

Install Ollama:

https://ollama.com/

Download a model:

```powershell
ollama pull qwen3:0.6b
```

Start a conversation:

```powershell
ollama run qwen3:0.6b
```

The model above is a small demonstration model, not a recommendation for high-quality teacher responses.

Choose a teacher based on its performance on your target tasks.

Check that the Ollama API is accessible:

```powershell
Invoke-RestMethod -Uri "http://localhost:11434/api/tags"
```

The default local API address is:

```text
http://localhost:11434
```

## 4. Step Two: Prepare Prompts

Create a `data` directory:

```powershell
New-Item -ItemType Directory -Force -Path .\examples\distillation\data
```

Create a prompt dataset:

```powershell
code .\examples\distillation\data\prompts.jsonl
```

Example contents:

```json
{"id":"prompt-001","task":"explanation","prompt":"Explain what a local language model is in simple terms."}
{"id":"prompt-002","task":"explanation","prompt":"Explain model quantization and why it is useful."}
{"id":"prompt-003","task":"coding","prompt":"Write a Python function that checks whether a number is prime."}
{"id":"prompt-004","task":"summarization","prompt":"Summarize the benefits of running language models locally."}
```

Each line must contain a valid JSON object with a non-empty `prompt` field.

You can add more examples for your intended application.

## 5. Step Three: Generate Teacher Responses

Script:

[`generate-teacher-data.py`](generate-teacher-data.py)

This script will:

1. Read prompts from a JSONL file.
2. Send each prompt to the local Ollama API.
3. Collect teacher responses.
4. Validate the returned content.
5. Save prompt-response examples to JSONL.

The generated records will use a consistent structure:

```json
{
  "id": "prompt-001",
  "task": "explanation",
  "prompt": "Explain what a local language model is in simple terms.",
  "response": "A local language model is a model that runs on your own computer or server.",
  "teacher_model": "qwen3:0.6b"
}
```

The actual dataset is stored as JSONL, with one record per line.

Example command:

```powershell
python .\examples\distillation\generate-teacher-data.py --model qwen3:0.6b --input .\examples\distillation\data\prompts.jsonl --output .\examples\distillation\data\generated.jsonl
```

The script will be developed to support these command-line options.

## 6. Step Four: Prepare Training Splits

Separate generated examples into:

| File | Purpose |
|---|---|
| `train.jsonl` | Used for student training. |
| `validation.jsonl` | Used to monitor training. |
| `test.jsonl` | Reserved for final evaluation. |

An illustrative split is 80% training, 10% validation, and 10% testing.

Use reproducible splitting and avoid placing duplicate or near-duplicate examples across different splits.

The training scripts will use the `prompt` and `response` fields from generated records.

## 7. Step Five: Train the Student

Script:

[`train-student.py`](train-student.py)

The training implementation will use:

- Hugging Face Transformers.
- TRL supervised fine-tuning.
- PEFT for LoRA adapters.
- Hugging Face Datasets.

The student will learn from teacher-generated prompt-response examples.

Example command:

```powershell
python .\examples\distillation\train-student.py --model YOUR_STUDENT_MODEL --train-file .\examples\distillation\data\train.jsonl --validation-file .\examples\distillation\data\validation.jsonl --output-dir .\examples\distillation\outputs\student-distilled
```

Replace `YOUR_STUDENT_MODEL` with a compatible pretrained Hugging Face model ID or local model path.

The implementation will document its supported training options and defaults.

**Important:** This example demonstrates response-based supervised fine-tuning, not direct logit-based distillation.

## 8. Step Six: Evaluate the Student

Script:

[`evaluate-student.py`](evaluate-student.py)

Evaluation should compare:

1. The original student model.
2. The trained student model.
3. The teacher model, where available.

Example evaluation command:

```powershell
python .\examples\distillation\evaluate-student.py --model YOUR_STUDENT_MODEL --adapter .\examples\distillation\outputs\student-distilled --test-file .\examples\distillation\data\test.jsonl
```

Evaluation should examine:

- Response relevance.
- Task correctness.
- Instruction following.
- Response quality.
- Inference speed.
- Memory usage.

Automated metrics alone cannot establish that a student has preserved all teacher capabilities. Use human review and trusted references where appropriate.

## 9. Training Approaches

The initial implementation focuses on response-based distillation using SFT and LoRA.

Additional approaches can be explored later:

| Method | Description |
|---|---|
| Response-based distillation | Learn from teacher-generated text. |
| Full fine-tuning | Update the student's model parameters. |
| LoRA | Train lightweight adapter parameters. |
| QLoRA | Train adapters with a supported quantized base model. |
| Logit-based distillation | Learn from teacher probability distributions. |

Advanced logit-based distillation requires access to teacher logits or equivalent probability information and is not implemented by the initial scripts.

## 10. Troubleshooting

### Ollama connection error

Check whether Ollama is running:

```powershell
ollama list
```

Check the local API:

```powershell
Invoke-RestMethod -Uri "http://localhost:11434/api/tags"
```

### Model download problems

Check the model name and available disk space.

```powershell
ollama list
```

### CUDA or GPU errors

Check the PyTorch CUDA installation:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

A CUDA-enabled PyTorch installation does not guarantee that every training library or quantization method will work on the same system.

### Out-of-memory errors

Try:

- A smaller student model.
- A shorter maximum sequence length.
- A smaller per-device batch size.
- Gradient accumulation.
- LoRA instead of full fine-tuning.
- Gradient checkpointing where supported.

See the [Hardware Guide](../../docs/hardware-guide.md) and [Troubleshooting Guide](../../docs/troubleshooting.md).

## 11. Reproducibility

For reproducible experiments, record:

- Teacher model and version.
- Student model and revision.
- Dataset version.
- Generation settings.
- Training configuration.
- Random seed.
- Python and library versions.
- Hardware information.

Keep training and evaluation datasets separate.

Review the licenses and usage terms of all models and datasets before redistribution.

## 12. Related Guides

Read the accompanying documentation:

- [Distillation Overview](../../guides/distillation/README.md)
- [Knowledge Distillation](../../guides/distillation/knowledge-distillation.md)
- [Teacher Model Selection](../../guides/distillation/teacher-model-selection.md)
- [Dataset Generation](../../guides/distillation/dataset-generation.md)
- [Student Model Training](../../guides/distillation/student-model-training.md)
- [Hardware Guide](../../docs/hardware-guide.md)
- [Quantization Guide](../../docs/quantization.md)

---

**Part of [Local LLM Cookbook](../../README.md)** — a practical, hardware-aware open-source resource for running, fine-tuning, quantizing, and deploying language models locally.