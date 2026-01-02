# Teaching Gemma to Think: Multi-Domain Reasoning with GRPO

**Subtitle:** Fine-tuning Gemma 3 1B for structured reasoning across 7 domains using Group Relative Policy Optimization

---

## Approach Overview

This submission tackles the challenge of teaching Gemma 3 1B to **show its work** across diverse domains. The goal is to produce consistent, structured reasoning output that works whether the task is mathematical, creative, or somewhere in between.

**Key Design Decisions:**

1. **GRPO over SFT**: Reinforcement learning enables the model to explore response variations and learn from reward signals, rather than just mimicking examples.

2. **Multi-Domain Strategy**: Training across 7 domains (math, coding, science, logic, creative writing, creative ideation, summarization) ensures the model generalizes its reasoning format.

3. **Format-First Reward**: Hard dependency on XML structure, where responses without proper `<reasoning>` and `<answer>` tags receive zero reward, forcing format compliance before any content evaluation.

4. **Domain-Aware Rewards**: Verifiable domains (math, coding) use correctness checking; creative domains use heuristic quality metrics (length, diversity, relevance).

### Architecture Overview

![Figure 1: Multi-Domain GRPO Training Pipeline - Data flows from 7 domains through DISCO sampling, into Gemma 3 1B with LoRA adapters, through the GRPO training loop with format-gated rewards, producing the final LoRA checkpoint.](https://raw.githubusercontent.com/vserifoglu/LLM_TEACHER/refs/heads/main/docs/Gemma_Reasoning_Fine_Tuning_Architecture.png?token=GHSAT0AAAAAADSDKDD5TZPCJDUWX7BCUJP22KX7PNA)

---

## Dataset Engineering

### Multi-Domain Dataset

| Domain | Source | Samples |
|--------|--------|---------|
| Math | GSM8K | ~2,500 |
| Coding | MBPP | ~900 |
| Science | SciQ, ARC | ~3,400 |
| Logic | StrategyQA | ~1,400 |
| Creative Writing | WritingPrompts | ~2,300 |
| Creative Ideation | Dolly, Longform | ~1,600 |
| Summarization | XSum | ~2,900 |

**Total: ~30,000 samples**

The full dataset preparation pipeline is available in this notebook: [Dataset Curation Pipeline](https://www.kaggle.com/code/fissalalsharef/dataset-curation-pipeline)


### DISCO Sampling

To prevent domain imbalance (science would dominate with 3,400 samples), we implemented **Distribution Correction (DISCO)** sampling:

```
P(domain) ∝ original_count^(1/temperature)
```

With temperature=0.5, this boosts underrepresented domains (coding: 913 → higher sampling rate) while moderating overrepresented ones.

---

## Training Strategy

### GRPO (Group Relative Policy Optimization)

GRPO improves on standard policy gradient methods by ranking responses within a group rather than comparing to a fixed baseline. For each prompt, we generate 4 candidate responses and compute relative rewards, allowing the model to learn from comparing better vs. worse attempts. See [GRPO Paper](https://arxiv.org/abs/2402.03300) for full mathematical derivation.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base Model | Gemma 3 1B IT | Competition requirement |
| LoRA Rank | 16 | Balance between capacity and memory |
| LoRA Alpha | 32.0 | 2x rank ratio |
| Learning Rate | 3e-6 | Conservative for stability |
| GRPO Generations | 4 | Group size for relative ranking |
| GRPO Iterations | 4 | Policy updates per batch |
| KL Penalty (β) | 0.04 | Prevent reward hacking |
| Generation Steps | 512 | Sufficient for complete responses |
| Temperature | 0.7 | Exploration during training |

### Reward Function Design

```
Total Reward = Format (gate) + Domain Reward

Format: Binary gate - must have valid XML structure
    ├── <reasoning>...</reasoning>
    ├── <answer>...</answer>
├── Correct order
└── Non-empty content

If format invalid → 0.0 (hard stop)

Domain Rewards:
├── Verifiable: 0.8 if correct, 0.0 if wrong
└── Creative: 0.3 length + 0.25 diversity + 0.25 relevance
```

### Key Insight: Generation Token Limit

**Critical discovery:** Initial training used 256 max tokens, but many responses need 300-500 tokens. Model was being cut off before closing `</answer>` tag, receiving 0.0 reward, and learning confused behavior.

**Fix:** Increased `TOTAL_GENERATION_STEPS` from 256 to 512.

---

## Prompt Design

### Template Structure

```python
SYSTEM_PROMPT = """Provide your reasoning in <reasoning> tags, then your final answer in <answer> tags.

Format:
<reasoning>Your step-by-step thinking</reasoning>
<answer>Your final answer</answer>"""

TEMPLATE = """<start_of_turn>user
{system_prompt}

Task: {question}<end_of_turn>
<start_of_turn>model"""
```

**Key Design Choices:**
- Clean formatting with no extra whitespace
- "Task:" prefix clearly separates instruction from question
- Example format in system prompt for clarity

### Critical Lesson: Prompt Matching

**Discovery:** Model performance dropped from 94% to 55% when inference prompt didn't match training prompt exactly, including whitespace!

The indentation in the template creates specific tokenization patterns. Inference prompts must match character-for-character.

---

## Tunix Experience

Tunix proved to be a capable framework for GRPO training on TPU:

**Challenges Encountered:**
- Documentation is still evolving
- Some trial-and-error with KV cache sizing
- Understanding the exact data format expected by the trainer

**Suggestions for Improvement:**
- More examples for multi-domain training scenarios
- Clearer guidance on memory optimization for longer sequences

---

## Results & Validation

> **[TO BE FILLED AFTER RETRAINING]**

### Format Compliance by Domain

> **[INSERT BAR CHART: Per-domain format compliance rates]**
> 
> Image file: `domain_compliance_chart.png`

| Domain | Format Compliance |
|--------|-------------------|
| Coding | ___% |
| Math | ___% |
| Science | ___% |
| Logic | ___% |
| Creative Writing | ___% |
| Creative Ideation | ___% |
| Summarization | ___% |
| **Overall** | **___%** |

### Prompt Matching Impact

> **[INSERT COMPARISON CHART: Before vs After prompt fix]**
>
> Shows the critical discovery: format compliance jumped from 55% to 94% when inference prompt matched training prompt.

| Prompt Version | Format Compliance |
|----------------|-------------------|
| Mismatched (wrong whitespace) | 55% |
| Matched (exact training format) | 94% |

### Sample Outputs

> **[INSERT 2-3 SAMPLE OUTPUTS showing reasoning quality]**

**Example 1 - Math:**
```
Question: [INSERT QUESTION]

<reasoning>[INSERT MODEL REASONING]</reasoning>
<answer>[INSERT ANSWER]</answer>
```

**Example 2 - Creative Writing:**
```
Task: [INSERT TASK]

<reasoning>[INSERT MODEL REASONING]</reasoning>
<answer>[INSERT CREATIVE OUTPUT]</answer>
```

**Example 3 - Coding:**
```
Task: [INSERT CODING TASK]

<reasoning>[INSERT MODEL REASONING]</reasoning>
<answer>[INSERT CODE]</answer>
```

---

## Challenges & Learnings

### 1. TPU Compute Constraints
- 9-hour session limit requires efficient training
- Careful hyperparameter selection to maximize value per step
- Checkpointing strategy to recover from failures

### 2. Prompt Format Sensitivity
- Discovered through validation that XML format compliance dropped 40% with modified prompt at inference time.

### 3. Token Limit Trap
- Short generation limit during training prevented model from completing responses
- Model learned confused behaviors from always getting truncated
- Fix: Generous token budget (512+) during training

### 4. Multi-Domain Balancing
- DISCO sampling essential for fair domain representation
- Without it, overrepresented domains (science) would dominate gradients

### What Would I Do Differently

- Start with higher generation token limit from day one
- Build validation pipeline earlier to catch prompt mismatches
- Test with greedy decoding (temp=0) earlier.

---

## References

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948)
- [Tunix Documentation](https://tunix.readthedocs.io/)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
