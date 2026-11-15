# Repository Structure

This document explains the organization of Local LLM Cookbook, the purpose of its directories, and where different types of documentation, examples, configurations, and scripts belong.

## Project Architecture

```text
local-llm-cookbook/
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug-report.yml
│   │   └── feature-request.yml
│   ├── workflows/
│   │   ├── markdown-lint.yml
│   │   └── python-checks.yml
│   ├── FUNDING.yml
│   └── PULL_REQUEST_TEMPLATE.md
│
├── assets/                  # Project images and visual assets
│
├── configs/
│   └── training/            # Reusable training configurations
│
├── docs/                    # Core project documentation
│   ├── README.md
│   ├── getting-started.md
│   ├── hardware-guide.md
│   ├── model-selection.md
│   ├── quantization.md
│   ├── glossary.md
│   └── troubleshooting.md
│
├── examples/
│   ├── datasets/            # Sample datasets
│   ├── inference/           # Local inference examples
│   ├── fine-tuning/         # Fine-tuning examples
│   └── deployment/          # Model deployment examples
│
├── guides/
│   ├── running-local-llms/  # Ollama and LM Studio
│   ├── fine-tuning/         # Unsloth and LLaMA-Factory
│   └── deployment/          # Model conversion and deployment
│
├── scripts/                 # Setup and utility scripts
│
├── tests/                   # Automated tests
│
├── .editorconfig
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── SECURITY.md
├── STRUCTURE.md
└── SUPPORT.md
```

## Directory Responsibilities

### `.github/`

Contains GitHub-specific configuration, issue templates, pull request templates, funding information, and automated CI workflows.

### `assets/`

Stores project branding and visual resources, including the repository logo and documentation images.

### `configs/`

Contains reusable configuration files for experiments, model fine-tuning, and training workflows.

### `docs/`

Contains foundational documentation covering hardware requirements, model selection, quantization, terminology, setup, and troubleshooting.

### `examples/`

Contains practical and reproducible code examples.

Each example should include relevant dependencies, execution instructions, and hardware assumptions where applicable.

### `guides/`

Contains detailed tutorials organized by workflow and tool.

Examples include running models with Ollama or LM Studio, fine-tuning with Unsloth or LLaMA-Factory, and exporting models to GGUF.

### `scripts/`

Contains utilities for environment setup, dependency checks, validation, and development workflows.

### `tests/`

Contains automated tests for example code, datasets, and configuration validation.

## Root-Level Files

| File | Purpose |
|---|---|
| `README.md` | Project overview and navigation |
| `STRUCTURE.md` | Repository architecture and organization |
| `LICENSE` | Open-source licensing terms |
| `CONTRIBUTING.md` | Contribution process |
| `CODE_OF_CONDUCT.md` | Community expectations |
| `SECURITY.md` | Vulnerability reporting |
| `SUPPORT.md` | Help and support information |
| `CHANGELOG.md` | Project change history |
| `pyproject.toml` | Python project configuration |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from Git |
| `.editorconfig` | Consistent editor settings |
| `.pre-commit-config.yaml` | Pre-commit checks |

## Organization Principles

The repository follows a few simple principles:

1. Keep foundational documentation separate from practical tutorials.
2. Keep runnable code inside `examples/`.
3. Keep reusable configurations inside `configs/`.
4. Organize guides by workflow rather than duplicating documentation.
5. Include clear execution instructions and hardware requirements.
6. Avoid unnecessary abstractions and premature complexity.
7. Prefer reproducible examples with clearly documented dependencies.

## Adding New Content

Before adding a new file, consider its purpose:

- General concepts and reference material belong in `docs/`.
- Step-by-step workflows belong in `guides/`.
- Runnable code belongs in `examples/`.
- Reusable settings belong in `configs/`.
- Development utilities belong in `scripts/`.
- Automated checks belong in `tests/`.

The goal is to keep Local LLM Cookbook easy to navigate, maintain, and extend as the project grows.