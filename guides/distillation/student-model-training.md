# Student Model Training

Student model training is the stage where a smaller language model learns from the teacher-generated dataset.

For this cookbook, the recommended beginner workflow is **response-based distillation using supervised fine-tuning (SFT)**.

This approach allows you to train a student using ordinary prompt-and-response examples without requiring access to the teacher's internal logits or hidden states.

This guide covers the training workflow, model preparation, training strategies, hardware considerations, evaluation, and deployment preparation.

## 1. Understand the Training Workflow

A typical response-based distillation workflow looks like this:

```text
       Teacher-Generated Dataset
                  |
                  v
        Dataset Validation
                  |
                  v
         Train / Validation Split
                  |
                  v
          Student Model
                  |
                  v
       Supervised Fine-Tuning
                  |
                  v
          Trained Student
                  |
                  v
        Evaluation and Testing
                  |
                  v
       Quantization / Deployment
```

The student model starts from an existing pretrained checkpoint. Training adapts its parameters using the prepared examples.

This is not training a language model from scratch.

## 2. Response-Based Distillation with SFT

Response-based distillation uses teacher-generated responses as supervised training targets.

For example:

**Training input:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Explain model quantization in simple terms."
    },
    {
      "role": "assistant",
      "content": "Model quantization reduces the numerical precision used to represent model weights or computations. This can reduce memory requirements and improve inference efficiency."
    }
  ]
}
```

The student learns to generate the assistant response when given the user prompt.

A common SFT objective is next-token prediction using cross-entropy loss over the target tokens.

Depending on the training configuration, loss can be calculated only over assistant responses rather than over the entire conversation.

**Important:** This is response-based distillation. It does not directly match the teacher's probability distributions. That requires a separate logit-based distillation objective.

## 3. Choose a Student Model

Select a pretrained model that can run within your hardware and deployment requirements.

Consider:

- Parameter count.
- Architecture.
- Tokenizer.
- Context length.
- Training framework compatibility.
- Model license.
- Supported precision and quantization methods.

Possible model families to explore include:

- Qwen
- Llama
- Gemma
- Mistral

Model availability, supported architectures, and license terms vary by checkpoint.

For response-based SFT, the student can use a different tokenizer from the teacher because it learns from text examples using its own tokenizer.

For logit-based distillation, teacher and student output vocabularies and token alignment require additional consideration.

## 4. Prepare the Training Dataset

Before training, ensure that the dataset follows the format required by your training framework.

A common conversational JSONL format:

```json
{"messages":[{"role":"user","content":"What is an LLM?"},{"role":"assistant","content":"An LLM is a language model trained on large amounts of text to understand and generate language."}]}
{"messages":[{"role":"user","content":"What is local inference?"},{"role":"assistant","content":"Local inference means running a model on your own hardware rather than relying on a remote inference service."}]}
```

The dataset should be divided into:

| Dataset | Purpose |
|---|---|
| Training | Updates student model parameters. |
| Validation | Monitors training and helps identify overfitting. |
| Test | Measures final performance on unseen examples. |

Do not use the test dataset to update model parameters or repeatedly tune training decisions.

### Apply the correct chat template

Different model families use different chat templates.

Use the student tokenizer's chat template when preparing conversational examples.

Do not assume that every model uses the same formatting tokens or conversation structure.

Also verify that the training framework correctly masks user and system tokens when your objective is to calculate loss only on assistant responses.

## 5. Choose a Training Strategy

There are several ways to train a student.

### 5.1 Full Fine-Tuning

Full fine-tuning updates the model's trainable parameters.

Advantages:

- Directly updates the entire model.
- Flexible training approach.
- Can work well with suitable data and compute.

Limitations:

- Higher memory requirements.
- More optimizer-state memory.
- More storage for checkpoints.
- Can be impractical on consumer hardware.

Full fine-tuning is generally not the best starting point for hardware-constrained experiments.

### 5.2 LoRA

Low-Rank Adaptation (LoRA) adds small trainable adapter matrices to selected model layers.

Instead of updating every original model parameter, training updates the adapter parameters.

Advantages:

- Fewer trainable parameters.
- Lower training memory requirements than full fine-tuning in many setups.
- Convenient adapter checkpoints.
- Useful for experimenting with multiple tasks.

### 5.3 QLoRA

QLoRA combines a quantized frozen base model with trainable LoRA adapters.

It can significantly reduce the memory required for fine-tuning larger models.

However, QLoRA still requires memory for activations, gradients, adapters, optimizer states, and runtime overhead.

A quantized base model does not mean training requires only the memory occupied by its weights.

### Comparison

| Strategy | Parameters updated | Typical resource demand |
|---|---|---|
| Full fine-tuning | Most or all model parameters | Highest |
| LoRA | Adapter parameters | Lower |
| QLoRA | Adapter parameters with quantized base weights | Often lower than LoRA with an unquantized base |

Actual requirements depend on architecture, sequence length, batch size, precision, optimizer, and framework implementation.

## 6. Recommended Training Tools

| Tool | Purpose |
|---|---|
| PyTorch | Core deep learning framework |
| Hugging Face Transformers | Load and work with pretrained models |
| Hugging Face Datasets | Load and prepare training data |
| TRL | Supervised fine-tuning and other post-training workflows |
| PEFT | LoRA and other parameter-efficient methods |
| bitsandbytes | Supported quantization and optimizer workflows |
| Unsloth | Optimized training workflows for supported models |

Choose a compatible combination of library versions and model architecture.

Do not assume every tool supports every model or quantization method.

## 7. Example Training Configuration

The following is an illustrative LoRA-based training configuration.

```yaml
model:
  name: "YOUR_STUDENT_MODEL"
  trust_remote_code: false

dataset:
  train_file: "data/train.jsonl"
  validation_file: "data/validation.jsonl"
  format: "conversational"

training:
  method: "sft"
  num_train_epochs: 3
  learning_rate: 0.0001
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_sequence_length: 1024
  warmup_ratio: 0.03

lora:
  enabled: true
  rank: 16
  alpha: 32
  dropout: 0.05

output:
  directory: "./outputs/student-distilled"
  save_strategy: "epoch"
```

This is a starting configuration, not a universal prescription.

The model name, dataset paths, training framework, precision, and supported configuration fields must be adapted to the selected implementation.

The effective batch size also depends on the number of distributed devices:

\[
B_{\text{effective}} =
B_{\text{device}}
\times G
\times N
\]

Where:

- \(B_{\text{device}}\) is the per-device batch size.
- \(G\) is the gradient accumulation count.
- \(N\) is the number of training devices.

## 8. Important Training Hyperparameters

| Parameter | Purpose |
|---|---|
| Learning rate | Controls the magnitude of parameter updates. |
| Epochs | Number of passes through the training dataset. |
| Batch size | Number of examples processed per device step. |
| Gradient accumulation | Combines gradients across multiple steps. |
| Sequence length | Maximum token length used for training examples. |
| Warmup | Gradually increases the learning rate at the start. |
| Weight decay | Regularization applied by supported optimizers. |
| LoRA rank | Controls the rank of the trainable adapter matrices. |

Start with conservative settings and monitor validation performance.

Increasing epochs or learning rate does not automatically improve the student.

## 9. Hardware Considerations

Training requirements depend on more than the model's parameter count.

Important factors include:

- Base model weight precision.
- Training strategy.
- Sequence length.
- Batch size.
- Gradient checkpointing.
- Optimizer states.
- Activation memory.
- GPU architecture and available VRAM.
- CPU RAM and storage.

### Practical hardware guidance

| Available GPU VRAM | Practical direction |
|---|---|
| 2–4 GB | Very constrained; focus on data preparation and small-model CPU experiments where feasible. |
| 6 GB | Explore very small models and carefully optimized configurations. |
| 8 GB | Some small-model LoRA or QLoRA experiments may be possible. |
| 12 GB | More room for supported small-model fine-tuning workflows. |
| 16–24 GB | Broader experimentation with larger student models and training configurations. |

These are approximate experimentation tiers, not guaranteed compatibility levels.

A 4 GB GPU should not be treated as a general-purpose 7B fine-tuning setup.

### Techniques for reducing memory usage

- Use LoRA or QLoRA where supported.
- Reduce sequence length.
- Reduce per-device batch size.
- Use gradient accumulation.
- Enable gradient checkpointing when appropriate.
- Use supported mixed-precision training.
- Avoid loading unnecessary model copies.
- Monitor GPU and system memory during training.

## 10. Monitor Training

Monitor both training and validation metrics.

Useful measurements include:

- Training loss.
- Validation loss.
- Learning rate.
- Training steps.
- GPU memory usage.
- Training time.
- Checkpoint size.

### Understanding loss

Training loss measures how well the model predicts the training targets.

Validation loss measures performance on held-out validation examples.

If training loss keeps decreasing while validation loss increases, the student may be overfitting.

A low training loss alone does not prove that the student has learned useful capabilities.

## 11. Save and Manage Checkpoints

Save checkpoints so that training can be resumed and earlier versions can be evaluated.

For LoRA training, the saved checkpoint may contain only adapter weights rather than a complete standalone model.

Keep track of:

- Base student model and revision.
- Tokenizer and chat template.
- Training dataset version.
- Training configuration.
- Library versions.
- Random seed.
- Checkpoint identifier.
- License information.

Do not overwrite the only copy of a useful checkpoint.

## 12. Evaluate the Student Model

After training, evaluate the student using examples that were not included in its training data.

Compare the student with:

1. The original pretrained student.
2. The teacher model.
3. The trained student.

Use the same prompts and evaluation conditions wherever possible.

Measure:

| Metric | What it tells you |
|---|---|
| Task accuracy | Whether the student completes the target task correctly. |
| Response quality | Whether answers are useful and relevant. |
| Instruction following | Whether the student follows user requirements. |
| Hallucination rate | How often unsupported information appears. |
| Tokens per second | Generation throughput. |
| Peak memory | Hardware requirements during inference. |
| Model size | Storage requirements. |

Use human review or trusted references for tasks where automatic metrics are insufficient.

A successful distillation experiment should meet its target quality requirements while offering a useful deployment advantage.

## 13. Advanced: Logit-Based Distillation

Logit-based distillation trains the student using probability distributions produced by the teacher.

A common objective combines supervised cross-entropy with a temperature-scaled KL-divergence loss.

\[
L =
\alpha L_{\text{CE}}
+
(1-\alpha)T^2 L_{\text{KD}}
\]

Where:

- \(L_{\text{CE}}\) is the supervised cross-entropy loss.
- \(L_{\text{KD}}\) is the teacher-student KL-divergence loss.
- \(T\) is the temperature.
- \(\alpha\) controls the contribution of supervised learning.

The exact formulation depends on the training implementation.

This approach requires teacher logits or probability distributions aligned with the student output vocabulary.

It is more advanced than ordinary response-based SFT and is outside the initial beginner training implementation.

## 14. Export and Prepare the Student for Deployment

After evaluating the trained model:

1. Preserve the original base model.
2. Save the trained adapter or model checkpoint.
3. Merge LoRA adapters when appropriate and supported.
4. Validate the merged model.
5. Convert to a supported deployment format if required.
6. Quantize only after checking model compatibility.
7. Test inference on the target hardware.

For GGUF workflows, see the [GGUF Export Guide](../deployment/gguf-export.md).

For quantization, see the [Quantization Guide](../../docs/quantization.md).

A successful training run does not automatically produce a deployment-ready model.

## 15. Training Checklist

- [ ] Select a compatible student model.
- [ ] Check model and dataset licenses.
- [ ] Validate training and validation datasets.
- [ ] Confirm the correct chat template.
- [ ] Choose full fine-tuning, LoRA, or QLoRA.
- [ ] Check hardware and memory requirements.
- [ ] Configure training hyperparameters.
- [ ] Run a small pilot experiment.
- [ ] Monitor training and validation loss.
- [ ] Save checkpoints and configuration.
- [ ] Evaluate on held-out test examples.
- [ ] Compare against the original student and teacher.
- [ ] Prepare and test the deployment artifact.

## 16. Next Steps

Continue with:

- [Knowledge Distillation](knowledge-distillation.md)
- [Teacher Model Selection](teacher-model-selection.md)
- [Dataset Generation](dataset-generation.md)
- [Distillation Examples](../../examples/distillation/README.md)

Related guides:

- [Hardware Guide](../../docs/hardware-guide.md)
- [Model Selection](../../docs/model-selection.md)
- [Quantization](../../docs/quantization.md)
- [Troubleshooting](../../docs/troubleshooting.md)

---

**Part of [Local LLM Cookbook](../../README.md)** — a practical, hardware-aware open-source resource for running, fine-tuning, quantizing, and deploying language models locally.