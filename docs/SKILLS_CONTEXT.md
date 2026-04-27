# Hugging Face Skills Context

This document serves as a quick reference for the Hugging Face skills available in this workspace. These skills empower the agent to perform advanced AI/ML tasks using the Hugging Face ecosystem.

## Available Skills

| Skill | Description | Path |
| :--- | :--- | :--- |
| **hf-cli** | Managing models, datasets, spaces, and jobs via `hf` CLI. | `.skills/huggingface/skills/hf-cli/SKILL.md` |
| **huggingface-llm-trainer** | Fine-tuning LLMs (SFT, DPO, GRPO) using TRL or Unsloth. | `.skills/huggingface/skills/huggingface-llm-trainer/SKILL.md` |
| **huggingface-datasets** | Fetching and processing datasets via Dataset Viewer API. | `.skills/huggingface/skills/huggingface-datasets/SKILL.md` |
| **huggingface-local-models** | Running GGUF models locally with llama.cpp. | `.skills/huggingface/skills/huggingface-local-models/SKILL.md` |
| **huggingface-gradio** | Building Python-based web UIs for ML demos. | `.skills/huggingface/skills/huggingface-gradio/SKILL.md` |
| **huggingface-trackio** | Experiment tracking and visualization. | `.skills/huggingface/skills/huggingface-trackio/SKILL.md` |
| **huggingface-vision-trainer** | Fine-tuning vision models (Detection, Classification, SAM). | `.skills/huggingface/skills/huggingface-vision-trainer/SKILL.md` |
| **transformers-js** | Running ML models in JS/TS (WASM/WebGPU). | `.skills/huggingface/skills/transformers-js/SKILL.md` |

## Usage Guidelines

- **Agent Instructions**: When performing a task related to any of the above, refer to the corresponding `SKILL.md` file for detailed workflows and commands.
- **Authentication**: Ensure a Hugging Face token is set in the environment or `.env` file for actions requiring write access.
- **Paths**: All skills are located in the `.skills/huggingface` directory.

---
> [!TIP]
> To verify the installation, you can run `huggingface-cli whoami` (if logged in) or check for available models with `huggingface-cli scan-cache`.
