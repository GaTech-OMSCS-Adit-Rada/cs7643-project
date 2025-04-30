import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

if 'google.colab' in sys.modules:
  print(f'Running in google colab. Our path is `{GOOGLE_DRIVE_PATH}`')
else:
  GOOGLE_DRIVE_PATH = '.'
  print('Running locally.')

# Get current working directory (sms_classification)
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Get parent directory (the root directory containing 'model')
parent_dir = os.path.dirname(current_dir)
print(f"Parent directory: {parent_dir}")

# Add parent directory to Python path
sys.path.append(parent_dir)
print(f"Python path now includes: {sys.path}")

################################

from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

################################

import torch
print("torch version:", version("torch"))

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device = " + device)
if device == 'cpu':
    print("WARNING: Using CPU will cause slower train times")

from torch.utils.data import Dataset, DataLoader
from torch import nn
import pandas as pd
import urllib
from pathlib import Path
import time
from tqdm import tqdm

################################

from model.gpt import GPTModel, text_to_token_ids, token_ids_to_text, generate_text_simple, generate, print_model_stats, TransformerBlock
from model.load_model import load_weights
from model.lora_gpt import LoRALayer, LinearWithLoRA, replace_linear_with_lora, replace_linear_with_lora_last_n

################################

from transformers import GPT2Model

# Available Models Names
model_names = {
    "gpt2-small (124M)": "openai-community/gpt2",
    "gpt2-medium (355M)": "openai-community/gpt2-medium",
    "gpt2-large (774M)": "openai-community/gpt2-large",
    "gpt2-xl (1558M)": "openai-community/gpt2-xl"
}

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

def get_raw_gpt(model_name):
    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}")

    base_config_copy = BASE_CONFIG.copy()
    base_config_copy.update(model_configs[model_name])
    return GPTModel(base_config_copy)

def get_pretrained_gpt_model(model_name, verbose=True):
    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}")

    base_config_copy = BASE_CONFIG.copy()
    base_config_copy.update(model_configs[model_name])
    gpt_model = GPTModel(base_config_copy)

    hf_pretrained_gpt = GPT2Model.from_pretrained(model_names[model_name], cache_dir="checkpoints")
    load_weights(gpt_model, hf_pretrained_gpt, base_config_copy)

    if verbose:
        print_model_stats(gpt_model, model_name)

    return gpt_model

################################

def convert_to_lora_model(model: GPTModel, rank: int, alpha: int, last_n_trf_blocks=None) -> GPTModel:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters before: {total_params:,}")

    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters after: {total_params:,}")

    if last_n_trf_blocks is not None:
        replace_linear_with_lora_last_n(model, n=last_n_trf_blocks, rank=rank, alpha=alpha)
    else:
        replace_linear_with_lora(model, rank=rank, alpha=alpha)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable LoRA parameters: {total_params:,}")
    model.to(device)
    return model

################################

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

torch.manual_seed(123)
test_raw_gpt = GPTModel(GPT_CONFIG_124M)
test_raw_gpt.eval()  # disable dropout

start_context = "Hello, I am"

tokenizer = tiktoken.get_encoding("gpt2")
encoded_tensor = text_to_token_ids(start_context, tokenizer)

print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
print("\nInput text:", start_context)
print("Encoded input text:", encoded_tensor)
print("encoded_tensor.shape:", encoded_tensor.shape)

out_token_ids = generate_text_simple(
    model=test_raw_gpt,
    token_ids=encoded_tensor,
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
decoded_text = token_ids_to_text(out_token_ids, tokenizer)

print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
print("\nOutput:", out_token_ids)
print("Output length:", len(out_token_ids[0]))
print("Output text:", decoded_text)


total_params = sum(p.numel() for p in test_raw_gpt.parameters())
print(f"Total Parameters: {total_params:,}")

total_params_gpt2 =  total_params - sum(p.numel() for p in test_raw_gpt.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")


print_model_stats(test_raw_gpt, "GPT-124M")

################################

# CHOOSE_MODEL = "gpt2-medium (355M)"
CHOOSE_MODEL = "gpt2-large (774M)"
test_pretrained_gpt = get_pretrained_gpt_model(CHOOSE_MODEL, verbose=False)

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=test_pretrained_gpt.to(device),
    # token_ids=text_to_token_ids("Every effort moves", tokenizer).to(device),
    token_ids=text_to_token_ids("The state capital of New Jersey is Newark. The state capital of California is", tokenizer).to(device),
    max_new_tokens=30,
    context_size=BASE_CONFIG["context_length"],
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

################################

test_lora_gpt = get_pretrained_gpt_model("gpt2-small (124M)", verbose=False)

total_params = sum(p.numel() for p in test_lora_gpt.parameters() if p.requires_grad)
print(f"Total trainable parameters before: {total_params:,}")

for param in test_lora_gpt.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in test_lora_gpt.parameters() if p.requires_grad)
print(f"Total trainable parameters after: {total_params:,}")

# replace_linear_with_lora(test_lora_gpt, rank=16, alpha=16)
replace_linear_with_lora_last_n(test_lora_gpt, n=2, rank=16, alpha=16)

total_params = sum(p.numel() for p in test_lora_gpt.parameters() if p.requires_grad)
print(f"Total trainable LoRA parameters: {total_params:,}")

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=test_lora_gpt.to(device),
    token_ids=text_to_token_ids("Every effort moves", tokenizer).to(device),
    max_new_tokens=30,
    context_size=BASE_CONFIG["context_length"],
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

################################

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

################################

# NEW CODE FOR CLASSIFICATION STARTS HERE

class SmsClassificationDataset(Dataset):
    def __init__(self, txt: str, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.labels = []

        # Use eos_token_id for padding
        self.pad_token_id = getattr(tokenizer, 'eos_token_id', 50256) # Default GPT2 eos

        # Parse the input string 'txt'
        for line in txt.strip().split('\n'):
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                label_str, text = parts
                self.labels.append(1 if label_str == "spam" else 0)
                self.texts.append(text)
            # else: Optionally handle malformed lines

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        print(f"\n--- Debugging __getitem__ for index {idx} ---")
        print(f"Original Text: '{text[:50]}...'") # Print first 50 chars

        # Tokenize
        encoded = self.tokenizer.encode(text)
        print(f"Encoded (len {len(encoded)}): {encoded[:10]}...{encoded[-10:]}") # Print first/last 10 tokens

        # Truncate if necessary, keeping the *last* self.max_length tokens
        original_encoded_len = len(encoded)
        if len(encoded) > self.max_length:
            print(f"Truncating: original len {original_encoded_len} > max_len {self.max_length}")
            # Calculate the starting index to keep the last max_length tokens
            start_index = len(encoded) - self.max_length
            encoded = encoded[start_index:]
            print(f"Encoded after truncation (len {len(encoded)}): {encoded}")
        else:
             print("No truncation needed.")

        input_ids_list = encoded # Keep potentially truncated sequence as list for now

        # Pad if necessary to the left
        if len(input_ids_list) < self.max_length:
            padding_len = self.max_length - len(input_ids_list)
            print(f"Padding: len {len(input_ids_list)} < max_len {self.max_length}. Adding {padding_len} pads.")
            # Use pad_token_id for padding
            input_ids_list = [self.pad_token_id] * padding_len + input_ids_list # Pad left
        else:
            print("No padding needed.")

        assert len(input_ids_list) == self.max_length, f"Length mismatch: {len(input_ids_list)} vs {self.max_length}"

        final_input_tensor = torch.tensor(input_ids_list)
        print(f"Final input_ids tensor: {final_input_tensor}")
        print(f"----------------------------------------")

        return final_input_tensor, torch.tensor(label, dtype=torch.float32)


def create_classification_dataloader(txt: str, batch_size=4, max_length=256,
                                     shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = SmsClassificationDataset(txt, tokenizer, max_length)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

# NEW CODE FOR CLASSIFICATION ENDS HERE

################################

sms_file_path = os.path.join(parent_dir, 'sms-spam.txt')

with open(sms_file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_classification_dataloader(
    raw_text, batch_size=1, max_length=4, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

################################

from sms_classification.train import calc_batch_loss, calc_loader_loss, evaluate_model, generate_and_print_sample, TrainingConfig, TrainingResults, train_model_simple, train_model
from sms_classification.train_plots import plot_losses, plot_perplexity, plot_and_save_learning_rate
import json

def save_training_results(results: TrainingResults, filename: str):
    results_dict = {
        "train_losses": results.train_losses,
        "val_losses": results.val_losses,
        "track_steps_seen": results.track_tokens_seen,
        "train_accuracies": results.train_accuracies,
        "val_accuracies": results.val_accuracies,
    }
    if results.track_lrs is not None:
        results_dict["track_lrs"] = results.track_lrs

    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=4)
        print(f"Training results saved to {filename}")

################################

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # Disable dropout during inference

with open(sms_file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

print(text_data[:99])

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))

print("Characters:", total_characters)
print("Tokens:", total_tokens)

# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


torch.manual_seed(123)

train_loader = create_classification_dataloader(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_classification_dataloader(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

################################

torch.manual_seed(123)
GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
training_config = TrainingConfig(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
)

sanity_check_training_results = train_model_simple(training_config)

# --- Save and Plot Results ---
results_filename = "sms_classification_results.json"
save_training_results(sanity_check_training_results, results_filename)
print(f"Results saved to {results_filename}")

plot_losses(training_config, sanity_check_training_results, filepath="sms_losses.png")

# Optional: Plot accuracy if function is available and updated
# Ensure plot_accuracy is defined in train_plots.py to handle steps
# try:
#     plot_accuracy(training_config.num_epochs, training_results.track_tokens_seen, # track_tokens_seen holds steps
#                   training_results.train_accuracies, training_results.val_accuracies,
#                   save_path="sms_accuracy.png")
# except NameError:
#     print("plot_accuracy function not found or not updated, skipping accuracy plot.")
# except Exception as e:
#     print(f"Error plotting accuracy: {e}")


################################
num_epochs = 15
print(len(val_loader))
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.2 * total_steps) # 20% warmup
print(warmup_steps)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

peak_lr = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)
tokenizer = tiktoken.get_encoding("gpt2")

num_epochs = 15
training_config = TrainingConfig(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="WINNER!! As a valued network customer",
    tokenizer=tokenizer,
    warmup_steps=warmup_steps,
    initial_lr=1e-5,
    min_lr=1e-5,
)

sanity_check_advanced_training_results = train_model(training_config)


################################

# Call plot_losses with the config and results objects
plot_losses(training_config, # Pass the TrainingConfig object
            sanity_check_advanced_training_results, # Pass the TrainingResults object
            filepath="3.pdf") # Pass the filepath directly

# The plotting function handles tight_layout and show/save
# plt.tight_layout(); plt.savefig("3.pdf")
# plt.show()

plt.figure(figsize=(7, 5))
plt.plot(range(len(sanity_check_advanced_training_results.track_lrs)), sanity_check_advanced_training_results.track_lrs)
plt.ylabel("Learning rate")
plt.xlabel("Steps")
plt.grid()
plt.show()

################################

# Renamed function to better reflect its purpose
def print_classification_metrics(model, train_loader, val_loader, device):
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        # Calculate Loss (using updated classification loss function)
        train_loss = calc_loader_loss(train_loader, model, device, num_batches=len(train_loader)) # Use full loader for better estimate
        val_loss = calc_loader_loss(val_loader, model, device, num_batches=len(val_loader))
        # Calculate Accuracy (using updated classification accuracy function)
        train_acc = calc_loader_accuracy(train_loader, model, device, num_batches=len(train_loader))
        val_acc = calc_loader_accuracy(val_loader, model, device, num_batches=len(val_loader))
    model.train() # Set back to train mode

    print(f"Training loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Training accuracy: {train_acc:.4f}") # Print accuracy
    print(f"Validation accuracy: {val_acc:.4f}") # Print accuracy
    # Removed perplexity calculations