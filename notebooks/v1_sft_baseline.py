"""
================================================================================
VARIANT 1: SFT BASELINE NOTEBOOK
================================================================================

Goal: Fine-tune Gemma 3 1B to produce the <reasoning>...</reasoning><answer>...</answer> format.

This notebook is structured in 6 sections:
1. Setup & Installs
2. Data Loading  
3. Data Preprocessing
4. Model Loading
5. Training
6. Save Checkpoint

Author: Learning project for understanding JAX/Tunix/Gemma fine-tuning
"""

# ============================================================================
# SECTION 1: SETUP & INSTALLS
# ============================================================================
# 
# WHY: We need to install the right versions of libraries that work together.
# Kaggle has pre-installed packages that can conflict with Tunix.
#
# Run this in a Kaggle notebook cell:
# !pip uninstall -q -y gensim bigframes tensorflow-decision-forests tf-keras flax jax jaxlib tensorflow
# !pip install -U -q google-cloud-storage google-cloud-automl protobuf
# !pip install -q "numpy>=2.0" "ml_dtypes>=0.4.0"
# !pip install -q "flax==0.12.0" "google-tunix[prod]==0.1.3"

# ============================================================================
# SECTION 1.1: IMPORTS
# ============================================================================

import os
import re
from pprint import pprint

# JAX - The foundation for everything
# WHY: JAX is like NumPy but with automatic differentiation and GPU/TPU support
import jax
import jax.numpy as jnp

# Flax - Neural network library built on JAX
# WHY: Flax provides the building blocks for defining neural networks
from flax import nnx

# Optax - Optimizer library
# WHY: Contains optimizers like AdamW that update model weights during training
import optax

# Orbax - Checkpointing library
# WHY: Saves and loads model weights to/from disk
from orbax import checkpoint as ocp

# Grain - Data loading library
# WHY: Efficiently loads and batches data for training
import grain

# Tunix - The main library for LLM fine-tuning
# WHY: Provides PeftTrainer for SFT and model loading utilities
from tunix.sft.peft_trainer import PeftTrainer, TrainingConfig, TrainingInput
from tunix.models.gemma3 import model, params

# CSV loading
import csv

print(f"JAX devices: {jax.devices()}")
print("Setup complete!")

# ============================================================================
# SECTION 1.2: HYPERPARAMETERS
# ============================================================================
#
# WHY: These control how training works. We keep them in one place so they're
# easy to find and change.

# ----- Data Settings -----
NUM_SAMPLES = 20        # 🧪 TESTING: Use only 20 samples. Change to None for full dataset.
BATCH_SIZE = 2          # How many examples to process at once

# ----- LoRA Settings -----
# WHY LoRA? Instead of updating ALL model weights (expensive), LoRA adds small
# trainable "adapter" layers. Much faster and uses less memory.
LORA_RANK = 64          # Size of the adapter layers
LORA_ALPHA = 64.0       # Scaling factor for LoRA

# ----- Training Settings -----
LEARNING_RATE = 3e-6    # How big of a step to take when updating weights
NUM_EPOCHS = 1          # How many times to go through the dataset
MAX_STEPS = 50          # 🧪 TESTING: Stop after 50 steps. Remove for full training.

# ----- Sequence Settings -----
MAX_SEQ_LENGTH = 512    # Maximum number of tokens in a sequence

# ----- TPU Sharding Settings -----
# WHY: TPUs have multiple cores. We need to tell JAX how to split the model.
# (1, 2) means 1 replica, 2-way tensor parallel
# "fsdp" = Fully Sharded Data Parallel, "tp" = Tensor Parallel
MESH_SHAPE = (1, 2)
MESH_AXIS_NAMES = ("fsdp", "tp")

# ----- Checkpoint Settings -----
CKPT_DIR = "/tmp/v1_checkpoints"
SAVE_EVERY_N_STEPS = 10  # 🧪 TESTING: Save frequently for debugging

print(f"Training config: {NUM_SAMPLES} samples, {NUM_EPOCHS} epochs, {MAX_STEPS} max steps")


# ============================================================================
# SECTION 2: DATA LOADING
# ============================================================================
#
# WHY: We need to get the GSM8K dataset which contains math problems with
# step-by-step solutions.
#
# The dataset is attached to Kaggle as CSV files:
#   /kaggle/input/grade-school-math-8k-q-a/main_train.csv
#   /kaggle/input/grade-school-math-8k-q-a/main_test.csv

# ----- Data Path -----
# WHY: Kaggle datasets are mounted at /kaggle/input/{dataset-name}
DATA_PATH = "/kaggle/input/grade-school-math-8k-q-a/main_train.csv"

def load_gsm8k(csv_path=DATA_PATH, num_samples=None):
    """
    Load the GSM8K dataset from CSV file.
    
    WHY this function: Encapsulates data loading so it's reusable and testable.
    WHY CSV: The dataset is attached to Kaggle as CSV, simpler than TFDS.
    
    Args:
        csv_path: Path to the CSV file
        num_samples: If set, only load this many samples (for testing)
    
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    print(f"Loading GSM8K from {csv_path}...")
    
    dataset = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # CSV has columns: 'question' and 'answer'
            dataset.append({
                'question': row['question'],
                'answer': row['answer'],
            })
            
            # Early exit if we only want a subset
            if num_samples and len(dataset) >= num_samples:
                break
    
    print(f"Loaded {len(dataset)} examples")
    return dataset


# ============================================================================
# SECTION 3: DATA PREPROCESSING
# ============================================================================
#
# WHY: The raw GSM8K data looks like this:
#   "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n...#### 72"
#
# We need to convert it to:
#   "<reasoning>Natalia sold 48/2 = 24 clips in May...</reasoning><answer>72</answer>"

# ----- Special Tokens -----
# WHY: These tags tell the model where reasoning and answers go
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

# ----- System Prompt -----
# WHY: Instructs the model on what format to produce
SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {REASONING_START} and \
{REASONING_END}. Then, provide the final answer (i.e., just one numerical \
value) between {ANSWER_START} and {ANSWER_END}."""

# ----- Chat Template -----
# WHY: Gemma expects a specific chat format with turn markers
TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model
{response}<end_of_turn>"""


def extract_answer_from_gsm8k(text):
    """
    Extract the final answer from GSM8K format.
    
    WHY: GSM8K uses "#### 72" to mark the final answer. We need to extract it.
    
    Example:
        Input:  "...sold 72 clips altogether.\\n#### 72"
        Output: "72"
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_reasoning_from_gsm8k(text):
    """
    Extract the reasoning steps from GSM8K format.
    
    WHY: The reasoning is everything before "####". We also clean up the
    calculator annotations like <<48/2=24>>.
    
    Example:
        Input:  "Natalia sold 48/2 = <<48/2=24>>24 clips\\n...\\n#### 72"
        Output: "Natalia sold 48/2 = 24 clips\\n..."
    """
    # Get everything before ####
    reasoning = text.split("####")[0].strip()
    
    # Remove calculator annotations like <<48/2=24>>
    # WHY: These are just for verification, not part of the natural reasoning
    reasoning = re.sub(r'<<[^>]+>>', '', reasoning)
    
    return reasoning


def format_example(question, answer):
    """
    Convert a single GSM8K example to our target format.
    
    WHY: This is the core transformation. We take raw data and create the
    exact format we want the model to learn.
    
    Args:
        question: The math problem
        answer: The GSM8K answer with reasoning and #### marker
    
    Returns:
        Formatted string ready for training
    """
    # Extract the parts
    reasoning = extract_reasoning_from_gsm8k(answer)
    final_answer = extract_answer_from_gsm8k(answer)
    
    # Build the model's response in our target format
    response = f"{REASONING_START}\n{reasoning}\n{REASONING_END}\n{ANSWER_START}{final_answer}{ANSWER_END}"
    
    # Put it all together using the chat template
    formatted = TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        question=question,
        response=response
    )
    
    return formatted


def preprocess_dataset(dataset):
    """
    Convert all examples in the dataset to training format.
    
    WHY: Apply our formatting to every example.
    """
    formatted_examples = []
    
    for example in dataset:
        formatted = format_example(example['question'], example['answer'])
        formatted_examples.append(formatted)
    
    print(f"Preprocessed {len(formatted_examples)} examples")
    return formatted_examples


# ============================================================================
# SECTION 4: MODEL LOADING
# ============================================================================
#
# WHY: We need to load the pre-trained Gemma 3 1B model and add LoRA adapters.
# The base model stays frozen; only the LoRA weights are trained.
#
# IMPORTANT: This uses a two-step process (from GRPO demo):
# 1. Load model → Save to intermediate checkpoint → Free memory
# 2. Reload with proper TPU sharding → Apply LoRA
#
# This avoids JAX sharding issues with LoRA.

import gc  # For garbage collection
import qwix  # LoRA utilities

# Intermediate checkpoint directory
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"


def save_initial_checkpoint():
    """
    STEP 4a: Load model and save to intermediate checkpoint.
    
    WHY two steps:
    - Direct LoRA application fails due to sharding issues
    - Saving to checkpoint first, then reloading with proper sharding works
    - This is the exact pattern from the GRPO demo
    """
    import os
    import shutil
    from tunix.models.gemma3 import params as gemma_params
    
    print("Step 4a: Saving model to intermediate checkpoint...")
    
    # Clean up any previous checkpoints
    if os.path.exists(INTERMEDIATE_CKPT_DIR):
        shutil.rmtree(INTERMEDIATE_CKPT_DIR)
    os.makedirs(INTERMEDIATE_CKPT_DIR, exist_ok=True)
    
    # Load model configuration
    model_config = model.ModelConfig.gemma3_1b()
    
    # Load the pre-trained model
    print("Loading Gemma 3 1B from Kaggle...")
    gemma = gemma_params.create_model_from_checkpoint(
        gemma_params.GEMMA3_1B_IT,
        model_config
    )
    
    # Create tokenizer (we'll need it later)
    tokenizer = gemma_params.create_tokenizer()
    
    # Save model state to intermediate checkpoint
    print(f"Saving to {INTERMEDIATE_CKPT_DIR}...")
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(gemma)
    checkpointer.save(os.path.join(INTERMEDIATE_CKPT_DIR, "state"), state)
    checkpointer.wait_until_finished()
    
    # Free memory - IMPORTANT for TPU
    del gemma
    del state
    gc.collect()
    
    print("Intermediate checkpoint saved!")
    return tokenizer, model_config


def load_model_with_sharding(model_config):
    """
    STEP 4b: Reload model from checkpoint with proper TPU sharding.
    
    WHY: Loading directly doesn't set up correct sharding for TPU.
    This approach uses JAX's sharding primitives properly.
    """
    from tunix.models.gemma3 import params as gemma_params
    import os
    
    print("Step 4b: Reloading model with TPU sharding...")
    
    # Create mesh for TPU distribution
    mesh = jax.make_mesh(MESH_SHAPE, MESH_AXIS_NAMES)
    print(f"Created mesh: {mesh}")
    
    # Create abstract model to get the shape/structure
    abs_gemma = nnx.eval_shape(
        lambda: gemma_params.create_model_from_checkpoint(
            gemma_params.GEMMA3_1B_IT, 
            model_config
        )
    )
    
    # Set up sharding for each parameter
    abs_state = nnx.state(abs_gemma)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )
    
    # Restore from checkpoint with proper sharding
    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(
        os.path.join(INTERMEDIATE_CKPT_DIR, "state"),
        target=abs_state
    )
    
    # Merge graph structure with restored parameters
    graph_def, _ = nnx.split(abs_gemma)
    gemma = nnx.merge(graph_def, restored_params)
    
    print("Model loaded with proper sharding!")
    return gemma, mesh


def apply_lora(base_model, mesh):
    """
    STEP 4c: Add LoRA adapters to the model.
    
    WHY LoRA:
    - Full fine-tuning updates ALL weights (billions of parameters)
    - LoRA adds small adapter layers (millions of parameters)
    - Same quality, much faster and less memory
    """
    print("Applying LoRA adapters...")
    
    # Define which layers get LoRA adapters
    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
            ".*attn_vec_einsum"
        ),
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
    )
    
    # Apply LoRA to the model
    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, **model_input
    )
    
    # Apply sharding constraints to LoRA parameters
    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)
    
    print(f"LoRA applied: rank={LORA_RANK}, alpha={LORA_ALPHA}")
    return lora_model


# ============================================================================
# SECTION 5: TRAINING
# ============================================================================
#
# WHY: This is where the actual learning happens. We show the model examples
# and update its weights to minimize prediction error.

def create_optimizer():
    """
    Create the AdamW optimizer with learning rate warmup.
    
    WHY AdamW:
    - Adam is a popular optimizer that adapts learning rate per-parameter
    - The "W" adds weight decay (regularization) to prevent overfitting
    
    WHY warmup:
    - Starting with high learning rate can destabilize training
    - Warmup gradually increases LR from 0 to target over first steps
    """
    # Create the learning rate schedule
    # WHY: Warmup for first 10%, then constant
    warmup_steps = int(MAX_STEPS * 0.1)
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=warmup_steps,
        decay_steps=MAX_STEPS,
        end_value=LEARNING_RATE * 0.1,
    )
    
    # Create the optimizer
    optimizer = optax.adamw(
        learning_rate=schedule,
        b1=0.9,      # Momentum for gradient
        b2=0.99,     # Momentum for squared gradient
        weight_decay=0.1,
    )
    
    # Add gradient clipping
    # WHY: Prevents exploding gradients that can crash training
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.1),
        optimizer,
    )
    
    print(f"Optimizer created: AdamW with LR={LEARNING_RATE}, warmup={warmup_steps} steps")
    return optimizer


def create_training_config():
    """
    Create the training configuration for PeftTrainer.
    
    WHY: Bundles all training settings into one object.
    """
    config = TrainingConfig(
        eval_every_n_steps=MAX_STEPS + 1,  # No eval during training (only at end)
        max_steps=MAX_STEPS,
        checkpoint_root_directory=CKPT_DIR,
    )
    
    print(f"Training config: max_steps={MAX_STEPS}, checkpoint_dir={CKPT_DIR}")
    return config


def tokenize_examples(examples, tokenizer):
    """
    Convert text examples to token IDs for training.
    
    WHY: Neural networks work with numbers, not text. We need to convert
    each character/word into a number (token ID).
    
    Args:
        examples: List of formatted text strings
        tokenizer: The Gemma tokenizer
    
    Returns:
        List of TrainingInput objects
    """
    training_inputs = []
    
    for text in examples:
        # Tokenize the text
        tokens = tokenizer.encode(text)
        
        # Truncate or pad to MAX_SEQ_LENGTH
        if len(tokens) > MAX_SEQ_LENGTH:
            tokens = tokens[:MAX_SEQ_LENGTH]
        else:
            # Pad with zeros
            tokens = tokens + [0] * (MAX_SEQ_LENGTH - len(tokens))
        
        # Create mask (1 for real tokens, 0 for padding)
        mask = [1.0 if t != 0 else 0.0 for t in tokens]
        
        # Convert to JAX arrays
        tokens_array = jnp.array(tokens, dtype=jnp.int32)
        mask_array = jnp.array(mask, dtype=jnp.float32)
        
        # Create TrainingInput
        training_inputs.append(TrainingInput(
            input_tokens=tokens_array,
            input_mask=mask_array,
        ))
    
    print(f"Tokenized {len(training_inputs)} examples to max length {MAX_SEQ_LENGTH}")
    return training_inputs


def run_training(model, optimizer, config, train_data):
    """
    Run the training loop.
    
    WHY: This ties everything together - model, optimizer, and data.
    The PeftTrainer handles the training loop internally.
    """
    print("Creating PeftTrainer...")
    
    trainer = PeftTrainer(
        model=model,
        optimizer=optimizer,
        training_config=config,
    )
    
    print("Starting training...")
    print(f"  - Dataset size: {len(train_data)}")
    print(f"  - Steps: {MAX_STEPS}")
    print()
    
    # Run training
    trainer.train(train_ds=train_data)
    
    print("Training complete!")
    return trainer


# ============================================================================
# SECTION 6: EVALUATION & CHECKPOINT
# ============================================================================
#
# WHY: After training, we want to verify the model learned the format
# and save the weights for V2.

def evaluate_format(model, tokenizer, test_questions):
    """
    Check if the model produces correct format.
    
    WHY: Our goal was to teach the format. Let's verify it works.
    """
    print("Evaluating format accuracy...")
    
    correct_format = 0
    
    for question in test_questions[:5]:  # Test on 5 examples
        # Generate response (simplified - real inference is more complex)
        prompt = TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=question,
            response=""  # Model fills this in
        )
        
        # TODO: Add actual generation code
        # For now, just check that model exists
        print(f"  Q: {question[:50]}...")
    
    print("Evaluation complete (placeholder)")


def save_checkpoint(trainer, path):
    """
    Save model weights for V2.
    
    WHY: V2 will load these weights and continue training with RL.
    """
    print(f"Saving checkpoint to {path}...")
    
    # Use Orbax to save
    checkpointer = ocp.StandardCheckpointer()
    state = nnx.state(trainer.model)
    checkpointer.save(path, state)
    checkpointer.wait_until_finished()
    
    print("Checkpoint saved!")


# ============================================================================
# MAIN: RUN EVERYTHING
# ============================================================================

def main():
    """
    Main function that runs the full pipeline.
    
    WHY: Organizes all the steps in order.
    """
    print("=" * 60)
    print("VARIANT 1: SFT BASELINE")
    print("=" * 60)
    print()
    
    # Step 1: Load data
    print("STEP 1: Loading data...")
    dataset = load_gsm8k(num_samples=NUM_SAMPLES)
    
    # Step 2: Preview raw data
    print("\nSample raw data:")
    print(f"  Question: {dataset[0]['question'][:80]}...")
    print(f"  Answer: {dataset[0]['answer'][:80]}...")
    
    # Step 3: Preprocess
    print("\nSTEP 3: Preprocessing data...")
    formatted_examples = preprocess_dataset(dataset)
    
    # Preview formatted data
    print("\nSample formatted data:")
    print(formatted_examples[0][:500])
    print("...")
    
    # Step 4: Load model (NOTE: This requires Kaggle TPU environment)
    print("\nSTEP 4: Loading model...")
    print("NOTE: Model loading requires Kaggle TPU. Skipping for local testing.")
    # gemma, tokenizer, config = load_gemma_model()
    # lora_model = apply_lora(gemma, mesh)
    
    # Step 5: Training (requires model)
    # print("\nSTEP 5: Training...")
    # optimizer = create_optimizer()
    # train_config = create_training_config()
    # train_data = tokenize_examples(formatted_examples, tokenizer)y<
    # trainer = run_training(lora_model, optimizer, train_config, train_data)
    
    # Step 6: Save checkpoint
    # print("\nSTEP 6: Saving checkpoint...")
    # save_checkpoint(trainer, CKPT_DIR + "/final")
    
    jls_extract_var = print
    jls_extract_var("\n" + "=" * 60)
    print("DONE! (Local test mode - model loading skipped)")
    print("=" * 60)


if __name__ == "__main__":
    main()
