"""Contains the Model Architecture for the GPT model."""

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)                        # Shape: (b, num_tokens, d_out) = (b, m, d_in) * (d_in, d_out)
        queries = self.W_query(x)                   # Shape: (b, num_tokens, d_out) = (b, m, d_in) * (d_in, d_out)
        values = self.W_value(x)                    # Shape: (b, num_tokens, d_out) = (b, m, d_in) * (d_in, d_out)

        # We implicitly split the matrices:
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # (b, num_heads, num_tokens, num_tokens) = (b, num_heads, num_tokens, head_dim) * (b, num_heads, head_dim, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Shape of attn_weights = (b, num_heads, num_tokens, num_tokens)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape of attn_weights @ values: (b, num_heads, num_tokens, num_tokens) * (b, num_heads, num_tokens, head_dim) = (b, num_heads, num_tokens, head_dim)
        # Shape of context_vec: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        # Shape of context_vec = (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Shape of context_vec: (b, num_tokens, d_out) = (b, num_tokens, d_out) * (d_out, d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
    

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """ x is of shape (batch_size, num_tokens, emb_dim) """
        mean = x.mean(dim=-1, keepdim=True)                     # Shape: (b, num_tokens, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)       # Shape: (b, num_tokens, 1)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)        # Shape: (b, num_tokens, emb_dim)
        return self.scale * norm_x + self.shift                 # Shape: (b, num_tokens, emb_dim)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """ x is of shape (batch_size, num_tokens, emb_dim) """
        return self.layers(x)       # Output Shape: (b, num_tokens, emb_dim)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.multi_head_att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x                                # (b, num_tokens, emb_dim)
        x = self.layer_norm1(x)                     # (b, num_tokens, emb_dim)
        x = self.multi_head_att(x)                  # (b, num_tokens, emb_dim)
        x = self.drop_shortcut(x)                   # (b, num_tokens, emb_dim)
        x = x + shortcut                            # (b, num_tokens, emb_dim)

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.layer_norm2(x)                     # (b, num_tokens, emb_dim)
        x = self.ff(x)                              # (b, num_tokens, emb_dim)
        x = self.drop_shortcut(x)                   # (b, num_tokens, emb_dim)
        x = x + shortcut                            # (b, num_tokens, emb_dim)

        return x                                    # (b, num_tokens, emb_dim)


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_layer_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, input_tok_ids):
        """ input_tok_ids is of shape (batch_size, num_tokens) """
        batch_size, seq_len = input_tok_ids.shape
        tok_embeds = self.tok_emb(input_tok_ids)                # Shape: (batch_size, num_tokens, emb_dim)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=input_tok_ids.device))
        x = tok_embeds + pos_embeds                             # Shape: (batch_size, num_tokens, emb_dim)
        x = self.drop_emb(x)                                    # Shape: (batch_size, num_tokens, emb_dim)
        x = self.trf_blocks(x)                                  # Shape: (batch_size, num_tokens, emb_dim)
        x = self.final_layer_norm(x)                            # Shape: (batch_size, num_tokens, emb_dim)
        logits = self.out_head(x)                               # Shape: (batch_size, num_tokens, vocab_size)
        return logits


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, token_ids, max_new_tokens, context_size):
    # token_ids is (b, T) array of indices in the current context

    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 10 tokens, and the context size is 15
        # then only the last 10 tokens are used as context
        model_input_token_ids = token_ids[:, -context_size:]

        with torch.no_grad():
            logits = model(model_input_token_ids)           # Shape: (b, num_tokens, vocab_size)

        # Focus only on the last time step
        # (b, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the id of the vocab entry with the highest logits value
        token_ids_next = torch.argmax(logits, dim=-1, keepdim=True)  # (b, 1)

        # Append sampled index to the running sequence
        token_ids = torch.cat((token_ids, token_ids_next), dim=1)  # (n, num_tokens+1)

    return token_ids


def generate(model, token_ids, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = token_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        token_ids = torch.cat((token_ids, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return token_ids


def print_model_stats(model: GPTModel, model_name: str):
    print("Model Name:", model_name)

    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\tTotal Parameters: {total_params:,}")

    # Print memory requirement
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"\tTotal Memory Requirement: {total_size_mb:.2f} MB")
