# V1 Notebook Execution Plan

## Overview
Create a Kaggle notebook that fine-tunes Gemma 3 1B on GSM8K using SFT to teach the `<reasoning>` format.

---

## Expected Outcomes

**Primary Goal**: Model produces correct XML format consistently.

| Metric | Target | Notes |
|--------|--------|-------|
| Format accuracy | >90% | Outputs have `<reasoning>` and `<answer>` tags |
| Correct tag order | >95% | `<reasoning>` appears before `<answer>` |
| Answer extractable | >90% | Can parse final number from `<answer>` |

**Not optimized in V1** (deferred to V2/V3):
- Answer correctness
- Reasoning quality

---

## Notebook Sections

### 1. Setup & Installs
- Install dependencies (reuse from `kaggle_install_commands.txt`)
- Import libraries
- Set hyperparameters

### 2. Data Loading
- Load GSM8K via `get_dataset()` from GRPO demo
- Preview raw data format

### 3. Data Preprocessing  
- Parse GSM8K answers → extract reasoning steps + final answer
- Reformat to XML: `<reasoning>...</reasoning><answer>N</answer>`
- Tokenize with Gemma tokenizer → `TrainingInput(tokens, mask)`

### 4. Model Loading
- Load Gemma 3 1B via `get_gemma_ref_model()`
- Apply LoRA via `get_lora_model()`

### 5. Training
- Create optimizer (AdamW with warmup)
- Create `TrainingConfig`
- Create `PeftTrainer`
- Run `trainer.train()`

### 6. Save Checkpoint
- Save final weights via Orbax
- Export as Kaggle Dataset for V2

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model | Gemma 3 1B | Faster iteration, matches reference |
| Training | LoRA (rank=64) | Memory efficient on TPU |
| Trainer | `PeftTrainer` | Tunix native SFT |
| Format | XML tags | Competition requirement |
| Training size | Minimal first (10-50 examples) | Debug code before scaling |
| Checkpointing | Every ~30 mins | Crash recovery |
| Evaluation | At end only | Simpler for debugging |

---

## Reusable from GRPO Demo

| Function | Use In |
|----------|--------|
| `get_dataset()` | Section 2 |
| `extract_hash_answer()` | Section 3 |
| `SYSTEM_PROMPT`, `TEMPLATE` | Section 3 |
| `get_gemma_ref_model()` | Section 4 |
| `get_lora_model()` | Section 4 |
| Hyperparameters block | Section 1 |
| Checkpoint saving code | Section 6 |

---

## New Code Needed

1. **Data Preprocessing Function**: Convert GSM8K → XML format
2. **Tokenization Pipeline**: Create `TrainingInput` from formatted text
3. **SFT Training Loop**: Replace GRPO with `PeftTrainer`

---

## 🧪 Testing Mode Settings (Change When Scaling)

These settings are for debugging only. Update before full training:

| Setting | Testing Value | Production Value |
|---------|---------------|------------------|
| `NUM_SAMPLES` | 10-50 | Full dataset (~7.5K) |
| `NUM_EPOCHS` | 1 | 1-3 (experiment) |
| `SAVE_INTERVAL` | Every 10 steps | Every ~30 mins |
| `EVAL_AT_END` | True | True (add during-training later) |
