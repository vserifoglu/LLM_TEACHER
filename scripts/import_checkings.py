from tunix.models.gemma3 import model, params

from tunix.generate.sampler import CacheConfig
from tunix.generate import sampler as sampler_lib
from tunix.generate.sampler import CacheConfig
import qwix
import jax
import nnx
from tunix.rl import base_rollout
from tunix.rl.algorithms.grpo import GRPOLearner, GRPOConfig

sampler = sampler_lib.Sampler(
    transformer=restored_model,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_SEQ_LENGTH + 256,  # room for generation
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)
# Test generation
test_q = "What is 3 + 5?"
prompt = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{test_q}<end_of_turn>\n<start_of_turn>model\n"
output = sampler.generate(prompt)
print(output)


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


verification_sampler = sampler_lib.Sampler(
    transformer=verification_policy,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=1624,  # 1024 + 600 for generation
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)
