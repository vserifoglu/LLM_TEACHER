## understanding LoRA.

import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        
        # 1. The "Frozen Giant" (Standard Linear Layer)
        # We simulate a pre-trained layer. In real life, this has existing weights.
        self.linear = nn.Linear(in_features, out_features)
        
        # Freeze the giant! (Stop gradient updates)
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False
        
        # 2. The LoRA Matrices (A and B)
        # Matrix A: Rank x In (Random Noise)
        # Matrix B: Out x Rank (Zeros)
        self.lora_a = nn.Parameter(torch.randn(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        
        # 3. The Scaling Factor
        self.scaling = alpha / rank
        
        # Initialize A with random, B with zeros (Standard LoRA recipe)
        # This ensures the adapter starts as "invisible"
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        # Path 1: The Frozen Giant
        # Standard matrix multiplication: Wx + bias
        result_frozen = self.linear(x)
        
        # Path 2: The LoRA Adapter
        # (x @ A.T) @ B.T -> We use .T because PyTorch stores weights as (out, in)
        # x is (batch_size, in_features)
        
        # Step 1: Down-project (x -> rank)
        lora_result = x @ self.lora_a.T
        
        # Step 2: Up-project (rank -> out)
        lora_result = lora_result @ self.lora_b.T
        
        # Step 3: Scale
        lora_result = lora_result * self.scaling
        
        # Path 3: Combine
        return result_frozen + lora_result

# --- LET'S TEST IT (The Napkin Math) ---

# 1. Setup our layer
# Input size 2, Output size 2 (Like our napkin matrix)
# Rank = 1, Alpha = 2
my_lora_layer = LoRALinear(in_features=2, out_features=2, rank=1, alpha=2)

# 2. Hack the weights to match our napkin example perfectly
# W = [[1, 2], [3, 4]]
with torch.no_grad():
    my_lora_layer.linear.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    my_lora_layer.linear.bias.fill_(0) # Ignore bias for this demo

# 3. Create Input x = [10, 20]
x = torch.tensor([[10.0, 20.0]])

# 4. BEFORE TRAINING (B is zeros)
# We expect the output to be [50, 110] (Original Model behavior)
print("Output Before Training (Should be [50, 110]):")
print(my_lora_layer(x)) 

# 5. SIMULATE TRAINING (Manually set A and B to learned values)
# A = [1, 2], B = [2, 3]
with torch.no_grad():
    my_lora_layer.lora_a.copy_(torch.tensor([[1.0, 2.0]]))
    my_lora_layer.lora_b.copy_(torch.tensor([[2.0], [3.0]])) # Shape needs to be (2,1) for column vector in math terms

print("\nOutput After Training (Should be [250, 410]):")
print(my_lora_layer(x))