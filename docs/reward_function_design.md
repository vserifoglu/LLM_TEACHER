# Reward Function Design Specification

## Overview

Industry-standard hybrid reward function for GRPO training across multi-domain reasoning tasks. Uses verifiable deterministic rewards for objective tasks and heuristic-based rewards for creative/subjective tasks.

---

## Design Principles

1. **Additive Components**: Sum multiple reward signals (0.2 + 0.2 + 0.2 + 0.2 = 1.0 max)
2. **Domain-Specific**: Different evaluation criteria per domain type
3. **Format-First**: Universal format compliance across all domains
4. **Stable Gradients**: Partial credit for incremental improvements
5. **Hard Format Dependency**: Invalid format → immediate 0.0 reward (early exit)

### **Competition Constraints**
- Max output tokens: <1K (reasoning + answer combined)
- TPU time: 9 hours/session, 20 hours/week
- Single session reproducibility required

---

## Domain Classification

### **Verifiable Domains** (Objective Ground Truth)
- Math
- Coding
- Science
- Logic

### **Creative Domains** (Subjective Quality)
- Creative Writing
- Creative Ideation
- Summarization
- Other/Unknown

---

## Reward Components

### **Universal Component (All Domains)**

#### 1. Format Compliance (0.2 points)
```python
# Strict XML validation:
# 1. Must have exactly ONE <reasoning> tag pair
# 2. Must have exactly ONE <answer> tag pair
# 3. <reasoning> must come BEFORE <answer>
# 4. Tags must be properly paired (no overlap or nesting)
# 5. Content must not be empty

if valid_xml_structure(response):
    reward += 0.2
else:
    return 0.0  # Hard stop - invalid format
```

**Invalid Examples:**
```xml
<!-- Missing tags -->
<reasoning>think</reasoning>
42

<!-- Wrong order -->
<answer>42</answer>
<reasoning>think</reasoning>

<!-- Multiple tags -->
<reasoning>a</reasoning><reasoning>b</reasoning>
<answer>42</answer>

<!-- Overlapping tags -->
<reasoning>think<answer>42</reasoning></answer>
```

**Valid Example:**
```xml
<reasoning>Step-by-step thinking here</reasoning>
<answer>Final answer here</answer>
```

**Rationale:** Strict validation ensures model learns exact formatting. No partial credit for "close enough."

---

### **Verifiable Domain Components (0.8 points)**

#### 2. Correctness (0.6 points)
```python
if domain in ["math", "science", "logic"]:
    if answer_matches_ground_truth(extracted_answer, ground_truth):
        reward += 0.6
```

**Ground Truth Matching:**
- **Math**: Normalize numbers, handle equivalent forms (0.5 = 1/2)
- **Science**: Exact string match (case-insensitive)
- **Logic**: Yes/No normalization

#### 3. Execution-Based (0.2 points) - Coding Only
```python
if domain == "coding":
    test_results = execute_with_tests(extracted_answer, test_cases)
    reward += 0.2 * (passed_tests / total_tests)
```

**Rationale:** Code must execute correctly. Partial credit for passing some tests.

---

### **Creative Domain Components (0.8 points)**

#### 2. Length Appropriateness (0.3 points)
```python
reasoning_len = len(reasoning.split())
answer_len = len(answer.split())

# Reasoning: 20-500 words
if 20 <= reasoning_len <= 500:
    reward += 0.15
else:
    reward += 0.15 * max(0, 1 - abs(reasoning_len - 250) / 500)

# Answer: 10-300 words
if 10 <= answer_len <= 300:
    reward += 0.15
else:
    reward += 0.15 * max(0, 1 - abs(answer_len - 150) / 300)
```

**Rationale:** 
- Prevents verbosity gaming
- Encourages substantial but concise responses
- Reflects competition's "clear reasoning" requirement

#### 3. Lexical Diversity (0.25 points)
```python
unique_words = len(set(answer.lower().split()))
total_words = len(answer.split())
diversity_score = unique_words / total_words

reward += 0.25 * diversity_score
```

**Rationale:** 
- Penalizes repetitive text
- Encourages rich vocabulary
- Simple proxy for quality

#### 4. Coherence/Relevance (0.25 points)
```python
# Extract key terms from prompt
prompt_terms = set(extract_keywords(prompt.lower()))
reasoning_terms = set(reasoning.lower().split())

# Calculate overlap
relevant_reasoning = len(prompt_terms & reasoning_terms) / len(prompt_terms)

reward += 0.25 * relevant_reasoning
```

**Rationale:**
- Ensures reasoning addresses the prompt
- Prevents off-topic responses
- Cheap alternative to semantic similarity

---

## Reward Function Pseudocode

```python
def compute_reward(domain, prompt, response, ground_truth=None, test_cases=None):
    # HARD DEPENDENCY: Strict XML validation with order checking
    if not valid_xml_structure(response):
        return 0.0  # Stop immediately - invalid format
    
    # Extract content (guaranteed to exist if validation passed)
    reasoning = extract_between_tags(response, "reasoning")
    answer = extract_between_tags(response, "answer")
    
    # Format is valid - start with format reward
    reward = 0.2
    
    # Domain-specific components
    if domain in ["math", "coding", "science", "logic"]:
        # VERIFIABLE: Correctness (0.6) + Execution (0.2)
        if is_correct(answer, ground_truth, domain):
            reward += 0.6
            
        if domain == "coding" and test_cases:
            reward += 0.2 * (tests_passed / total_tests)
        else:
            reward += 0.2  # Full credit if correct
    
    else:
        # CREATIVE: Length (0.3) + Diversity (0.25) + Coherence (0.25)
        reward += 0.15 * length_score(reasoning, target=250, range=(20, 500))
        reward += 0.15 * length_score(answer, target=150, range=(10, 300))
        reward += 0.25 * lexical_diversity(answer)
        reward += 0.25 * prompt_relevance(prompt, reasoning)
    
    return reward  # Max: 1.0
```

---

## Helper Functions to Implement

```python
# Strict XML validation (order + uniqueness + pairing)
valid_xml_structure(response) -> bool
    # 1. Exactly ONE <reasoning>...</reasoning> pair
    # 2. Exactly ONE <answer>...</answer> pair
    # 3. <reasoning> appears BEFORE <answer>
    # 4. No overlapping/nested tags
    # 5. Content not empty (after strip)

# Format extraction (only call after validation passes)
extract_between_tags(text, tag_name) -> str

# Correctness checking (domain-specific normalization)
is_correct(answer, ground_truth, domain) -> bool
    # Math: normalize_numbers(answer) == normalize_numbers(truth)
    # Logic: normalize_yes_no(answer) == normalize_yes_no(truth)
    # Science: case_insensitive_match(answer, truth)

# Length scoring (smooth decay outside range)
length_score(text, target, range) -> float [0.0-1.0]
    # 1.0 if within range, decay linearly outside

# Lexical diversity
lexical_diversity(text) -> float [0.0-1.0]
    # unique_words / total_words

# Prompt relevance
prompt_relevance(prompt, reasoning) -> float [0.0-1.0]
    # keyword_overlap(prompt, reasoning)
```

---

## Design Rationale

### Why Early Exit on Invalid Format?

**Hard Dependency Pattern:**
```python
if not valid_format:
    return 0.0  # Stop immediately
```

**Rationale:**
1. **Non-Negotiable Requirement**: Competition requires exact XML format
2. **Clear Learning Signal**: "Fix format first, then we evaluate quality"
3. **Fast Convergence**: Models learn format in <500 steps
4. **Prevents Gaming**: High-quality wrong-format answers get no reward
5. **Computational Efficiency**: Skip expensive checks if format is invalid

**Industry Standard:**
- InstructGPT uses hard format requirements
- Clear hierarchy: Format → Correctness → Quality
- Format is foundation - everything else builds on it

**Alternative Considered:**
```python
# Soft dependency (NOT recommended)
reward = 0.0
if valid_format:
    reward += 0.2
reward += quality_score()  # Always computed
```

**Why Rejected:**
- Confusing signal (model might learn wrong-format-but-good-content)
- Wasted computation
- Doesn't align with competition's strict format requirement

---

### Why Additive Components?

**Problem with Binary Rewards:**
```python
# Bad: All or nothing
reward = 1.0 if perfect else 0.0
```

**Gradient Issues:**
- Model gets zero signal from "almost correct" attempts
- Cannot learn incremental improvements

**Solution: Additive Components**
```python
# Good: Partial credit
reward = format_reward + correctness_reward + quality_reward
```

**Benefits:**
- Model learns from partial success
- Smoother optimization landscape
- Easier to debug (can see which component is failing)

---

### Why Different Rewards for Creative vs Verifiable?

**Verifiable domains have objective answers:**
- Math: 2 + 2 = 4 (not 3.9)
- Coding: Tests pass or fail
- Clear binary signal

**Creative domains have no "correct" answer:**
- Creative writing: Many valid stories
- Summarization: Multiple good summaries
- Need quality proxies

---

## Testing Strategy

### Unit Tests

```python
def test_format_reward():
    valid = "<reasoning>think</reasoning><answer>42</answer>"
    invalid = "<reasoning>think</reasoning>42"
    
    assert compute_reward("math", "", valid, "42") >= 0.2
    assert compute_reward("math", "", invalid, "42") == 0.0

def test_math_correctness():
    correct = "<reasoning>2+2=4</reasoning><answer>4</answer>"
    wrong = "<reasoning>2+2=5</reasoning><answer>5</answer>"
    
    r1 = compute_reward("math", "2+2=?", correct, "4")
    r2 = compute_reward("math", "2+2=?", wrong, "4")
    
    assert r1 > r2
    assert r1 >= 0.8  # Format + correctness

def test_creative_length():
    short = "<reasoning>ok</reasoning><answer>yes</answer>"
    good = "<reasoning>" + " ".join(["word"]*100) + "</reasoning><answer>" + " ".join(["word"]*50) + "</answer>"
    
    r1 = compute_reward("creative_writing", "prompt", short)
    r2 = compute_reward("creative_writing", "prompt", good)
    
    assert r2 > r1
```

---

## Performance Considerations

**Speed Requirements:**
- Must compute ~1000 rewards/second (for GRPO batches)
- All operations are O(n) where n = text length
- No external API calls
- No neural network inference

**Memory:**
- Stateless function (no caching needed)
- Processes one sample at a time
- Minimal memory footprint

---

## Competition Alignment

### Evaluation Criteria Mapping

| Competition Metric | Our Reward Component |
|-------------------|---------------------|
| "Correct final answer" | Correctness (0.6) |
| "Clear reasoning trace" | Length appropriateness (0.3) |
| "Proper format" | Format compliance (0.2) |
| "Quality of reasoning" | Coherence + Diversity (0.5) |

**Coverage:** Our reward function addresses all stated eval criteria.

---

## Implementation Checklist

- [ ] Implement `extract_tag_content()`
- [ ] Implement `is_correct()` with domain normalization
- [ ] Implement `length_reward()`
- [ ] Implement `compute_diversity()`
- [ ] Implement `compute_coherence()`
- [ ] Implement main `compute_reward()` function
- [ ] Write unit tests
- [ ] Test on sample data from each domain
- [ ] Verify speed (>1000 rewards/sec)
- [ ] Integrate with GRPO trainer

---

## References

- **InstructGPT Paper**: Used length + diversity heuristics initially
- **RLHF Best Practices**: Verifiable rewards > learned rewards when possible
- **Hugging Face TRL GRPO**: Supports custom reward functions
- **Competition Discussion**: Additive components recommended by participants

---

## Future Improvements (Post-Competition)

- Train small reward model on human preferences
- Add semantic similarity for creative domains
- Implement self-consistency checks
- Domain-adaptive reward weighting
