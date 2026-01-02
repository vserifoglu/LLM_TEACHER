# Teaching Gemma to Think: Google Tunix Hackathon Submission

**Competition:** [Google Tunix Hackathon: Training Reasoning Models with JAX](https://www.kaggle.com/competitions/google-tunix-hackathon)

## Project Overview

This project implements a **single-session multi-domain reasoning training pipeline** for the Google Tunix Hackathon. We fine-tune **Gemma 3 1B** to "show its work" by generating explicit reasoning traces (`<reasoning>...</reasoning>`) before providing a final answer (`<answer>...</answer>`).

Unlike approaches that focus solely on mathematical reasoning, our model is trained to reason across **7 distinct domains** simultaneously, using a custom reward function that adapts to both verifiable (objective) and creative (subjective) tasks.

## Strategy: The "Generalist" Reasoner

We use **Group Relative Policy Optimization (GRPO)** to train the model in a single 9-hour TPU session.

- **Model:** Gemma 3 1B (Instruction Tuned)
- **Framework:** JAX / Tunix
- **Hardware:** Kaggle TPU v5e-8
- **Training Method:** GRPO with LoRA (Rank 16)

### Unique Features

1.  **Multi-Domain Training**: Simultaneous training on Math, Coding, Science, Logic, Creative Writing, Creative Ideation, and Summarization.
2.  **DISCO Sampling**: "Distribution Correction" sampling to balance dataset representation (boosting underrepresented domains like Coding).
3.  **Hybrid Reward Function**:
    *   **Format Gate**: Strict XML structure requirement (0.0 reward if invalid).
    *   **Verifiable Rewards**: Correctness checks for Math, Science, Coding.
    *   **Creative Rewards**: Rubric-based heuristics (Length, Diversity, Relevance) for creative tasks.

## Architecture

![Architecture](https://raw.githubusercontent.com/vserifoglu/LLM_TEACHER/refs/heads/main/docs/Gemma_Reasoning_Fine_Tuning_Architecture.png?token=GHSAT0AAAAAADSDKDD5TZPCJDUWX7BCUJP22KX7PNA)

The pipeline flows from data curation through DISCO sampling into the GRPO training loop, where the model generates 4 candidate responses per prompt. These are evaluated by our hybrid reward function to update the LoRA adapter policy.

## Dataset

We curated a dataset of **~30,000 samples** from diverse sources:
*   **Math**: GSM8K
*   **Coding**: MBPP
*   **Science**: SciQ, AI2_ARC
*   **Logic**: StrategyQA
*   **Creative**: WritingPrompts, Dolly-15k, XSum

**Data Pipeline Notebook:** [Dataset Curation Pipeline](https://www.kaggle.com/code/fissalalsharef/dataset-curation-pipeline)

## How to Run

1.  **Environment Needs**: Kaggle TPU v5e-8 session.
2.  **Notebook**: The training and evaluation are contained in a single notebook for the competition submission.
3.  **Hyperparameters**:
    *   `TOTAL_GENERATION_STEPS`: 512 (Crucial for complete reasoning traces)
    *   `KV_CACHE_SIZE`: Tuned for performance
    *   `TEMPERATURE`: 0.7 (Training) / 0.0 (Evaluation)

## Results

The model achieves high format compliance (>90%) across all domains and demonstrates the ability to adapt its reasoning style—proving that a small 1B model can learn generalized structured thinking.