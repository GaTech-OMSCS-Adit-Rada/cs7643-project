"""Contains modules to create LoRA layers for the GPT model."""

import torch
import torch.nn as nn
import math
from model.gpt import GPTModel, TransformerBlock


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)


def _apply_lora_to_module(module, rank, alpha):
    """Recursively replaces nn.Linear layers with LinearWithLoRA within a module."""
    for name, child_module in module.named_children():
        if isinstance(child_module, nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            # Make sure the original linear layer's weights are preserved
            original_linear = child_module
            lora_linear = LinearWithLoRA(original_linear, rank, alpha)
            setattr(module, name, lora_linear)
            # print(f"Replaced {name} in {module.__class__.__name__} with LoRA.") # Optional: for debugging
        else:
            # Recursively apply the same function to child modules
            _apply_lora_to_module(child_module, rank, alpha)


def replace_linear_with_lora_last_n(model: GPTModel, n: int, rank: int, alpha: float):
    """
    Replaces nn.Linear layers with LinearWithLoRA layers only in the last 'n'
    TransformerBlocks of the GPTModel.

    Args:
        model (GPTModel): The GPT model instance.
        n (int): The number of final transformer blocks to modify.
        rank (int): The rank of the LoRA decomposition.
        alpha (float): The alpha scaling factor for LoRA.
    """
    if not isinstance(model, GPTModel):
        raise TypeError("Model must be an instance of GPTModel")
    if not hasattr(model, 'trf_blocks') or not isinstance(model.trf_blocks, nn.Sequential):
         raise ValueError("Model does not have the expected 'trf_blocks' Sequential module.")

    num_total_blocks = len(model.trf_blocks)

    if n <= 0:
        print("Warning: n <= 0. No LoRA layers will be added.")
        return
    if n > num_total_blocks:
         print(f"Warning: n ({n}) is greater than the total number of transformer blocks ({num_total_blocks}). Applying LoRA to all transformer blocks.")
         n = num_total_blocks # Or raise an error if preferred

    # Calculate the starting index of the blocks to modify
    start_index = num_total_blocks - n

    print(f"Applying LoRA with rank={rank}, alpha={alpha} to the last {n} transformer blocks (indices {start_index} to {num_total_blocks - 1}).")

    # Iterate through only the last n blocks
    for i in range(start_index, num_total_blocks):
        block_to_modify = model.trf_blocks[i]
        if isinstance(block_to_modify, TransformerBlock):
             # Apply the recursive replacement function *only* to this block
            _apply_lora_to_module(block_to_modify, rank, alpha)
        else:
            print(f"Warning: Expected TransformerBlock at index {i}, but found {type(block_to_modify)}. Skipping.")
    
    # Apply lora to out head
    model.out_head = LinearWithLoRA(model.out_head, rank, alpha)
