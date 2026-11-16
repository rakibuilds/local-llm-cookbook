
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SOURCE = DATA_DIR / "generated-v3.jsonl"
OUTPUT = DATA_DIR / "reviewed-v3.jsonl"

# Human-reviewed replacement responses.
# Original IDs, tasks, prompts and teacher_model values are preserved.
REVIEWED_RESPONSES = {
    "001": """A local LLM is a language model whose inference runs on hardware you control, such as your own laptop, desktop or server. You do not need to train the model yourself: you can download an existing pretrained model and run it locally.

Local inference means that prompts are processed by your local runtime rather than being sent to a remote inference provider. The model files and runtime are stored or executed locally, although downloading models, checking for updates or using optional online features may still require internet access.

Training and inference are different operations. Inference uses an existing model to generate outputs. Training changes model parameters using data. A model can be trained in the cloud and later downloaded for local inference.""",

    "002": """A typical decoder-only large language model development lifecycle includes several possible stages:

1. Data preparation: collect, filter, deduplicate and prepare training data.
2. Pretraining: train the model, commonly with a causal next-token prediction objective, so it learns to predict the next token from preceding tokens.
3. Supervised fine-tuning (SFT): train on curated examples, such as instruction-and-response pairs, to improve instruction following.
4. Alignment or preference optimization: optionally use preference data or other techniques to encourage desirable behavior.
5. Evaluation: test capabilities, reliability, safety and performance.
6. Optimization and packaging: optionally quantize or otherwise optimize the model for deployment.
7. Deployment: serve the model through a local runtime, application or API.

This is a conceptual lifecycle, not a mandatory sequence for every user. Someone running an existing model locally usually only needs to download a compatible model, configure a runtime and evaluate it for their use case. They do not need to pretrain or fine-tune it.

Many decoder-only LLMs use causal language modeling. Masked-language modeling is a different objective associated with models such as BERT.""",

    "003": """A base model is a pretrained language model that has learned statistical patterns from its training data. It is commonly trained with a next-token prediction objective in which it predicts the next token from the preceding context. A base model can continue text, but it may not reliably follow instructions in the conversational way users expect.

An instruction-tuned model is further trained on examples of instructions and suitable responses. This helps it respond to requests, answer questions and follow conversational directions.

Instruction tuning does not necessarily make a model an expert in one specific subject. Its behavior depends on its base training, fine-tuning data, architecture and other training stages. Both base and instruction-tuned models can be run locally if their formats, sizes and requirements are supported by the chosen runtime.""",

    "004": """Model parameters are the learned numerical values inside a neural network. A model with more parameters generally requires more memory to store its weights, but parameter count alone does not determine quality, speed or capability.

RAM is system memory used by the operating system, applications and CPU-based workloads. VRAM is memory on a discrete GPU. When running a model on a GPU, its weights and other runtime data generally need to fit in available VRAM, although some runtimes support partial CPU offloading.

Weight storage depends on parameter count and numerical precision. As a rough illustration, one billion parameters stored at FP16 require about 2 GB for weights alone. Actual memory use is higher because of runtime buffers, activations, the KV cache and other overhead.

Context length also matters: longer conversations can increase KV-cache memory. Quantization can reduce weight storage, but the final memory requirement depends on the model, quantization format, runtime and workload.""",

    "005": """An 8 GB RAM, CPU-only computer can run some small quantized language models, but the practical limit depends on the operating system, available free memory, CPU, model architecture, quantization and context length.

Start with a small model and a conservative context size. Check the model's documented memory requirements and leave memory available for the operating system and other applications. A model's file size is not its complete runtime memory requirement: inference also needs memory for the KV cache, buffers and other data.

Use a lightweight runtime such as llama.cpp or Ollama with a compatible quantized model. Begin with a short prompt and a modest context window, then monitor memory use and generation speed. If the system starts swapping heavily or becomes unresponsive, use a smaller model, a more compact quantization or a shorter context.

There is no universal model-size or token-count guarantee for every 8 GB computer. Measure performance on the actual machine.""",

    "006": """CPU, GPU and Apple Silicon are different hardware options for local inference, and their performance cannot be ranked reliably without knowing the exact device, model and runtime.

A CPU can run quantized models and is useful when no supported accelerator is available, but generation may be slower. A CUDA-compatible NVIDIA GPU can accelerate supported workloads, especially when enough VRAM is available for the model and runtime data. Apple Silicon uses unified memory shared by the CPU and GPU, and supported runtimes can use its GPU acceleration.

Performance depends on architecture, memory bandwidth, available memory, quantization, context length, batch size, runtime implementation and accelerator support. Prompt processing and token generation can also have different performance characteristics.

Compare systems using the same model, quantization, context and runtime where possible. Measure prompt-processing speed, generation speed in tokens per second, time to first token and peak memory use. Avoid treating a single latency estimate as universal.""",

    "007": """Ollama, LM Studio and llama.cpp can all be used for local LLM inference, but they serve different roles.

Ollama provides a local model-management and inference service with a command-line interface and local HTTP API. It makes it convenient to download, run and integrate supported models into applications.

LM Studio is a desktop application for discovering, downloading and running compatible local models. It provides a graphical interface and can expose a local API for applications.

llama.cpp is a lightweight C/C++ inference project and library designed to run supported language models across different hardware backends. It is widely used with GGUF model files and offers command-line and server capabilities.

These tools are not simply cloud platforms. Their local inference capabilities allow prompts to be processed on the user's machine. Model-format and architecture support varies by runtime and version, so check compatibility before downloading a model.""",

    "008": """A context window is the maximum amount of token context a model can process in a single request. It generally includes the input prompt and the tokens generated in the response, subject to the model and runtime's limits.

The input can include system instructions, conversation history, user messages, tool descriptions and other formatting tokens. Therefore, the space available for a new response is smaller when the input already uses much of the context window.

A longer context window does not automatically mean better answers. The model must support that context length, and the runtime must have enough memory and compute. In many transformer models, the KV cache stores attention-related information for previous tokens and can consume substantial memory as context grows.

Applications may truncate, summarize or otherwise manage conversation history when a request exceeds the available context. The exact behavior depends on the application and runtime.""",

    "009": """Temperature, top-k and top-p are sampling controls that influence how a language model selects its next token.

Temperature scales the model's logits before sampling. Lower temperature makes the probability distribution more concentrated, while higher temperature generally makes it flatter and can increase variation. Temperature does not guarantee factual accuracy or confidence.

Top-k limits sampling to the k tokens with the highest probabilities. A smaller k restricts the candidate set; a larger k allows more candidates.

Top-p, also called nucleus sampling, selects the smallest set of highest-probability tokens whose cumulative probability reaches the chosen p threshold. A lower p generally narrows the candidate set, while a higher p generally includes more candidates.

These controls interact with one another and with the model and prompt. Their effects are probabilistic, not guarantees of creativity, correctness or consistency.""",

    "010": """For raw weight storage, approximate memory per parameter is:

- FP32: 4 bytes per parameter.
- FP16: 2 bytes per parameter.
- INT8: about 1 byte per parameter.
- 4-bit representation: about 0.5 bytes per parameter.

For one billion parameters, the idealized weight storage would therefore be approximately 4 GB at FP32, 2 GB at FP16, 1 GB at INT8 and 0.5 GB at 4-bit.

These are simplified estimates, not guaranteed model file sizes or total runtime memory requirements. Real quantized formats use metadata, scales and block structures. Runtime memory also includes buffers, activations, the KV cache and other overhead.

FP16 is a floating-point format with lower precision and storage than FP32. INT8 and 4-bit formats use lower-bit representations, often with quantization schemes designed to preserve model quality. Quantization can reduce memory requirements, but quality and speed depend on the model, quantization method and hardware.""",

    "011": """Post-training quantization (PTQ) converts a model to a lower-precision representation after its training or fine-tuning has finished. Some PTQ methods use calibration data to estimate activation ranges or other quantization parameters. Calibration does not necessarily update the model through gradient-based training.

Quantization-aware training (QAT) simulates quantization effects during training or fine-tuning. The model is optimized while accounting for the expected quantized representation, which can help it adapt to quantization-related errors.

PTQ is often simpler and less expensive because it does not require another training stage. QAT requires a training or fine-tuning process and suitable data and compute. Neither method guarantees a particular quality or speed improvement; results depend on the model, quantization approach, hardware and evaluation task.""",

    "012": """GGUF stands for GPT-Generated Unified Format. It is a binary file format used to store language models and associated metadata, and it is commonly used by llama.cpp and compatible inference tools.

A GGUF file can contain model tensors, metadata and tokenizer information needed by a compatible runtime. GGUF is not a TensorFlow format.

Quantization names describe how model tensors are represented. Q8_0 is an 8-bit quantization format. Q4_K_M is a 4-bit K-quantization variant that uses mixed precision for selected tensors; it should not be understood as every weight and bias simply being stored at exactly four bits.

Quantized GGUF files can reduce storage and memory requirements compared with higher-precision weights. However, smaller files do not automatically mean faster inference or better output quality. The result depends on hardware, runtime support, memory bandwidth, model architecture and the chosen quantization.""",

    "013": """Full fine-tuning, LoRA and QLoRA are different ways to adapt a pretrained model.

In full fine-tuning, the training process updates the model's original trainable parameters. This can require substantial GPU memory and compute.

LoRA (Low-Rank Adaptation) freezes the original model weights and adds trainable low-rank adapter matrices to selected layers. Training updates these adapters rather than a small selected portion of the original weights. The adapters are much smaller than the full model in many configurations.

QLoRA loads a quantized, typically 4-bit, frozen base model and trains LoRA adapters on top of it. The base weights remain quantized and frozen; the trainable adapter parameters are handled separately and are not simply the quantized base model being updated.

LoRA and QLoRA can reduce fine-tuning memory requirements, but they still need suitable compute, memory, data preparation and compatible software. The best method depends on the task, model and available hardware.""",

    "014": """Supervised fine-tuning (SFT) trains a pretrained model on curated examples, commonly consisting of instructions and target responses. The objective is to make the model produce responses that match the training examples.

Data quality is critical. Examples should be relevant, accurate, consistently formatted and appropriate for the intended task. Remove or resolve corrupted records, review duplicates, protect personal information and check that examples do not contain misleading or unsafe content.

Create training, validation and test splits before training. Keep related or duplicated examples in the same split where appropriate to reduce leakage. Use validation data for model selection and reserve test data for final evaluation.

For decoder-only language models, use a training framework that correctly handles tokenization, padding, attention masks and labels. Hugging Face Transformers with TRL and PEFT are commonly used tools for SFT and adapter-based fine-tuning. Follow the selected model and library documentation rather than treating a simple linear layer or generic DataLoader example as a complete LLM fine-tuning implementation.

Avoid blindly lowercasing text, removing punctuation or deleting special tokens: these operations can change meaning or break model formatting. Data augmentation should be reviewed for quality and must not be used as a substitute for removing leakage or correcting inaccurate examples.""",

    "015": """Knowledge distillation transfers useful behavior or information from a teacher model to a student model. The teacher is generally the model providing targets or guidance; the student is the model being trained.

The student does not have to use the same architecture as the teacher. It may be smaller or otherwise different, provided the training method and target information are compatible.

In response-based distillation, prompts are sent to the teacher and its generated answers are collected as training targets. The student can then be trained on prompt-and-response examples, often using supervised fine-tuning.

Other distillation methods can transfer teacher probability distributions, internal representations or relationships between examples. These methods require different kinds of access and training objectives.

A smaller student may be cheaper to run, but this is not guaranteed to preserve the teacher's quality. Evaluate the student on held-out tasks and measure its quality, latency and memory use on the intended hardware.""",

    "016": """Knowledge distillation methods differ in what information is transferred from the teacher to the student.

Response-based distillation uses teacher-generated outputs as targets. For example, a teacher can answer a set of prompts and the student can be trained on those prompt-and-answer pairs using supervised fine-tuning. This is closely related to ordinary SFT; the distinctive feature is that the target responses were generated or selected from a teacher.

Logit-based distillation transfers the teacher's probability distribution over possible next tokens, often using temperature-scaled softmax distributions and a divergence loss such as KL divergence. Access to teacher logits or equivalent token probabilities is required.

Feature-based distillation transfers intermediate representations, such as hidden states or selected layer activations. This generally requires access to the teacher's internal representations.

Relation-based distillation transfers relationships among examples, tokens, layers or representations, such as similarity or distance structures.

Response-based distillation can work with ordinary text outputs from a model API. Logit- and feature-based methods usually require a model interface that explicitly exposes the relevant information. A softmax distribution represents probabilities over tokens; it is not itself the generated text response.""",

    "017": """A practical black-box, response-based distillation workflow can be built using a teacher model that exposes text generation, even when its internal logits and hidden states are unavailable.

1. Define the student's intended tasks and evaluation criteria.
2. Create a diverse set of prompts that represent those tasks.
3. Generate teacher responses using the teacher's supported interface.
4. Store prompts and responses with useful metadata, such as example ID, task, teacher identifier and generation settings.
5. Validate the dataset structure, remove malformed records, inspect duplicates and review response quality.
6. Split the data into training, validation and test sets before training. Keep the test set separate from training and model selection.
7. Fine-tune a compatible student model on the training prompt-and-response examples.
8. Evaluate the student on held-out prompts using task-appropriate metrics, human review where appropriate, and performance measurements.

A normal text-generation API does not automatically provide teacher logits, attention maps or hidden states. Use those methods only if the provider explicitly exposes the required data. Repeated API sampling may produce alternative responses, but it does not provide internal teacher information.

Check the teacher model or API terms, output-use restrictions, the student base model license, dataset rights and privacy obligations. Do not include sensitive personal information without a valid basis and appropriate safeguards. Dataset validation checks data structure and quality; student evaluation measures how well the trained model performs. They are separate activities.""",

    "018": """A prime number is an integer greater than 1 that has exactly two positive divisors: 1 and itself. Zero, one and negative integers are not prime.

A reliable Python implementation is:

def is_prime(n):
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")

    if n < 2:
        return False

    if n % 2 == 0:
        return n == 2

    divisor = 3
    while divisor * divisor <= n:
        if n % divisor == 0:
            return False
        divisor += 2

    return True

The explicit bool check matters because bool is a subclass of int in Python. The loop only checks odd divisors and stops when divisor squared exceeds n, avoiding an unnecessary floating-point square-root conversion.

Examples:
- is_prime(2) returns True.
- is_prime(17) returns True.
- is_prime(1) returns False.
- is_prime(-5) returns False.
- is_prime(21) returns False.""",

    "019": """A JSONL validator should parse each nonblank line independently and verify that each record is a JSON object with the required fields. It should not assume that every JSON value is an object or that every field must be a string.

Example using only the Python standard library:

import json

REQUIRED_FIELDS = ("id", "prompt", "response")

def validate_jsonl(path):
    errors = []
    seen_ids = set()
    valid_records = 0

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"Line {line_number}: invalid JSON: {exc.msg}"
                )
                continue

            if not isinstance(record, dict):
                errors.append(
                    f"Line {line_number}: record must be a JSON object"
                )
                continue

            missing = [
                field for field in REQUIRED_FIELDS
                if field not in record
            ]
            if missing:
                errors.append(
                    f"Line {line_number}: missing fields: "
                    + ", ".join(missing)
                )
                continue

            invalid = [
                field for field in REQUIRED_FIELDS
                if not isinstance(record[field], str)
                or not record[field].strip()
            ]
            if invalid:
                errors.append(
                    f"Line {line_number}: fields must be non-empty "
                    f"strings: {', '.join(invalid)}"
                )
                continue

            record_id = record["id"]
            if record_id in seen_ids:
                errors.append(
                    f"Line {line_number}: duplicate id: {record_id}"
                )
                continue

            seen_ids.add(record_id)
            valid_records += 1

    return valid_records, errors

The validator skips blank lines, checks the top-level record type, checks required fields individually, rejects whitespace-only values and detects duplicate IDs. A production validator can add schema-specific type checks, duplicate-prompt detection, reporting and optional output of cleaned records.""",

    "020": """Evaluate a student model on data that was not used for training. Create training, validation and test splits before training, and keep the test set separate from training and model selection.

Choose metrics that match the task. Classification accuracy is the proportion of predictions that are correct:

accuracy = (true_positives + true_negatives) / total_examples

For imbalanced classification, also consider precision, recall, F1 and per-class metrics. Balanced accuracy is a different metric; for binary classification it is commonly the average of sensitivity and specificity.

For language generation, use task-appropriate measures. Perplexity measures how well a language model assigns likelihood to a sequence of tokens and requires token-level probabilities from the model. Text-overlap metrics such as BLEU or ROUGE can be useful for some tasks, but they do not establish factual correctness or overall answer quality. Human evaluation and task-specific tests may be necessary.

Compare training and validation behavior to look for overfitting. A widening gap between training and validation performance can be a warning sign, but there is no universal overfitting score obtained by dividing validation accuracy by test accuracy. Do not repeatedly tune against the test set.

Also measure deployment performance under the same conditions for teacher and student: time to first token, prompt-processing speed, generation speed in tokens per second, peak RAM or VRAM use and, where relevant, energy or cost. Use the same prompts, context settings, hardware and runtime configuration for a fair comparison.

A successful student should be judged on task quality and practical deployment requirements, not merely on how closely its answers resemble the teacher."""
}


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE}"
        )

    if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
        raise FileExistsError(
            f"{OUTPUT.name} already contains data. "
            "Move it aside or empty it before running this script."
        )

    records = []
    with SOURCE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in source line {line_number}: {exc}"
                ) from exc

        expected_ids = {f"{number:03d}" for number in range(1, 21)}

    def normalize_id(value):
        value = str(value)
        if value.startswith("prompt-"):
            return value.removeprefix("prompt-")
        return value

    actual_ids = {
        normalize_id(record.get("id", ""))
        for record in records
    }

    if len(records) != 20 or actual_ids != expected_ids:
        raise ValueError(
            "Expected exactly 20 source records with IDs prompt-001 "
            "through prompt-020. "
            f"Found {len(records)} records and IDs: "
            f"{sorted(actual_ids)}"
        )

    if set(REVIEWED_RESPONSES) != expected_ids:
        raise ValueError(
            "The reviewed response IDs do not match the expected IDs."
        )

    output_lines = []
    for record in records:
        record_id = normalize_id(record["id"])
        record["response"] = REVIEWED_RESPONSES[record_id]
        output_lines.append(
            json.dumps(record, ensure_ascii=False)
        )

    with OUTPUT.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(output_lines) + "\n")

    print(f"Created: {OUTPUT}")
    print(f"Reviewed records: {len(output_lines)}")
    print("Original source file was not modified.")


if __name__ == "__main__":
    main()
