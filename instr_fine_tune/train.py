"""Contains functions to train the model for instruction fine-tuning."""

import math
import torch
from dataclasses import dataclass
from model.gpt import generate_text_simple, text_to_token_ids, token_ids_to_text

def calc_batch_loss(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)     # Shape: (b, num_tokens)
    logits = model(input_batch)                                                     # Shape: (b, num_tokens, vocab_size)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loader_loss(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_batch_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loader_loss(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loader_loss(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, token_ids=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print('\t', decoded_text.replace("\n", " "))  # Compact print format
    model.train()


@dataclass
class TrainingConfig:
    model: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    optimizer: torch.optim.Optimizer
    device: torch.device
    num_epochs: int
    eval_freq: int
    eval_iter: int
    start_context: str
    tokenizer: object
    warmup_steps: int = None
    initial_lr: float = 3e-05
    min_lr: float = 1e-6
    run_name: str = "test"

@dataclass
class TrainingResults:
    train_losses: list
    val_losses: list
    track_tokens_seen: list
    train_perplexity: list
    val_perplexity: list
    track_lrs: list = None


def train_model_simple(config: TrainingConfig):
    train_losses, val_losses, track_tokens_seen = [], [], []
    train_perplexity, val_perplexity = [], []
    tokens_seen, global_step = 0, -1

    # 1) Iterate over training epochs
    for epoch in range(config.num_epochs):
        config.model.train()

        # 2) Iterate over batches
        for input_batch, target_batch in config.train_loader:
            # 3) Reset loss gradients from previous batch iteration
            config.optimizer.zero_grad()

            # 4) Calculate loss on current batch
            loss = calc_batch_loss(input_batch, target_batch, config.model, config.device)

            # 5) Backward pass to calculate loss gradients
            loss.backward()

            # 6) Update model weights using loss gradients
            config.optimizer.step()

            # 7a) Logging
            tokens_seen += input_batch.numel()
            global_step += 1

            # 7b) Optional evaluation step
            if global_step % config.eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    config.model, config.train_loader, config.val_loader, config.device, config.eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_perplexity.append(torch.exp(torch.tensor(train_loss)).item())
                val_perplexity.append(torch.exp(torch.tensor(val_loss)).item())
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f} "
                      f"Train perplexity {train_perplexity[-1]:.3f}, Val perplexity {val_perplexity[-1]:.3f}")

        # 8) Print a sample text after each epoch
        generate_and_print_sample(
            config.model, config.tokenizer, config.device, config.start_context
        )

    return TrainingResults(
        train_losses=train_losses,
        val_losses=val_losses,
        track_tokens_seen=track_tokens_seen,
        train_perplexity=train_perplexity,
        val_perplexity=val_perplexity
    )

def train_model(config: TrainingConfig):
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    train_perplexity, val_perplexity = [], []
    tokens_seen, global_step = 0, -1

    # Retrieve the initial l.r from the optimizer, assuming we use it as the peak l.r
    peak_lr = config.optimizer.param_groups[0]["lr"]

    # Calculate the total number of steps in the training process
    total_training_steps = len(config.train_loader) * config.num_epochs

    # Calculate the learning rate increment during the warmup phase
    lr_increment = (peak_lr - config.initial_lr) / config.warmup_steps

    # 1) Iterate over training epochs
    for epoch in range(config.num_epochs):
        config.model.train()

        # 2) Iterate over batches
        for input_batch, target_batch in config.train_loader:
            # 3) Reset loss gradients from previous batch iteration
            config.optimizer.zero_grad()
            global_step += 1

            # 4) Adjust the learning rate based on the current phase (warmup or cosine annealing)
            if global_step < config.warmup_steps:
                # 4a) Linear warmup
                lr = config.initial_lr + global_step * lr_increment
            else:
                # 4b) Cosine annealing after warmup
                progress = ((global_step - config.warmup_steps) /
                            (total_training_steps - config.warmup_steps))
                lr = config.min_lr + (peak_lr - config.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            # 4c) Apply the calculated learning rate to the optimizer
            for param_group in config.optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # Store the current learning rate

            # 5) Calculate and backpropagate the loss to fill grad
            loss = calc_batch_loss(input_batch, target_batch, config.model, config.device)
            loss.backward()

            # 6) Apply gradient clipping after the warmup phase to avoid exploding gradients
            if global_step >= config.warmup_steps:
                torch.nn.utils.clip_grad_norm_(config.model.parameters(), max_norm=1.0)

            # 7) Update model weights using clipped gradients
            config.optimizer.step()
            tokens_seen += input_batch.numel()

            # 8) Periodically evaluate the model on the training and validation sets
            if global_step % config.eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    config.model, config.train_loader, config.val_loader,
                    config.device, config.eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_perplexity.append(torch.exp(torch.tensor(train_loss)).item())
                val_perplexity.append(torch.exp(torch.tensor(val_loss)).item())
                track_tokens_seen.append(tokens_seen)
                # Print the current losses
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}, "
                      f"Train perplexity {train_perplexity[-1]:.3f}, "
                      f"Val perplexity {val_perplexity[-1]:.3f}, "
                      f"LR {lr:.3e}"
                      )

        # 9) Generate and print a sample from the model to monitor progress
        generate_and_print_sample(
            config.model, config.tokenizer, config.device, config.start_context
        )

    return TrainingResults(
        train_losses=train_losses,
        val_losses=val_losses,
        track_tokens_seen=track_tokens_seen,
        train_perplexity=train_perplexity,
        val_perplexity=val_perplexity,
        track_lrs=track_lrs
    )

