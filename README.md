# Google Tunix Hack: Training Reasoning Models with JAX

## Project Overview

This project fine-tunes Google's Gemma 2B model to "show its work" by generating explicit reasoning traces (`<reasoning>...</reasoning>`) before providing a final answer. We utilize Tunix (JAX-native library) on Kaggle TPUs to implement a multi-stage training pipeline that bridges verifiable (Math) and non-verifiable (Creative/Logic) domains.

## Strategy & Roadmap

We adopt a 3-Variant "Relay Race" Strategy to maximize the 9-hour TPU session limit and prevent catastrophic forgetting.

1. **Variant 1 (The Baseline):** Supervised Fine-Tuning (SFT) on GSM8K to teach the XML `<reasoning>` format.
2. **Variant 2 (The Specialist):** Reinforcement Learning (GRPO) on Math data using the V1 weights. Optimizes for correctness.
3. **Variant 3 (The Generalist):** Transfer Learning on a Mixed Dataset (80% Creative/Logic + 20% Math) using V2 weights. Uses a custom Router Reward Function to apply different scoring rubrics based on the task type.

## Architecture

The pipeline is designed as a serial chain of TPU sessions, where the output weights of one session become the input for the next.

![Architecture](https://raw.githubusercontent.com/vserifoglu/LLM_TEACHER/refs/heads/main/docs/fine_tuning_pipeline.png?token=GHSAT0AAAAAADQHJLSM7NPKJHKYN725S4QY2J7CJVA)

### Workflow Steps

- **Session A (SFT):** Training gemma-2b → Outputs `weights_v1` (Dataset Artifact).
- **Session B (Math RL):** Loads `weights_v1` → Applies GRPO (Math Rewards) → Outputs `weights_v2`.
- **Session C (Mixed RL):** Loads `weights_v2` → Applies GRPO (Router Rewards) → Outputs `final_submission`.

## Technical Stack

- **Base Model:** Gemma 2B (Instruction Tuned)
- **Framework:** JAX / Tunix
- **Hardware:** Kaggle TPU v5e-8
- **Algorithm:** GRPO (Group Relative Policy Optimization)
- **Key Feature:** Custom "Router" Reward Function (Switching between binary math correctness and soft rubric scoring).

## How to Run

1. **Notebook 1:** Run the SFT baseline. Save output to Kaggle Datasets.
2. **Notebook 2:** Attach Dataset 1. Run GRPO (Math). Save output.
3. **Notebook 3:** Attach Dataset 2. Run GRPO (Mixed). Submit the final model ID.