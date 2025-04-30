"""Contains functions to train the model for instruction fine-tuning."""

import math
import torch
from dataclasses import dataclass
from model.gpt import generate_text_simple, text_to_token_ids, token_ids_to_text

def calc_batch_loss(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loader_loss(data_loader, model, device, num_batches=None):
    """Calculates average classification loss over a dataloader."""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_batch_loss(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def calc_loader_accuracy(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

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
    train_accuracies: list
    val_accuracies: list
    track_lrs: list = None


def train_model_simple(config: TrainingConfig):
    # Initialize lists to store metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], [] # Added accuracy lists
    track_tokens_seen = [] # Track tokens instead of steps
    tokens_seen, global_step = 0, -1

    # 1) Iterate over training epochs
    for epoch in range(config.num_epochs):
        config.model.train() # Set model to training mode

        # 2) Iterate over batches
        for input_batch, target_batch in config.train_loader:
            # 3) Reset loss gradients from previous batch iteration
            config.optimizer.zero_grad()

            # 4) Calculate loss on current batch (using classification loss)
            loss = calc_batch_loss(input_batch, target_batch, config.model, config.device)

            # 5) Backward pass to calculate loss gradients
            loss.backward()

            # 6) Update model weights using loss gradients
            config.optimizer.step()

            # 7a) Logging step count
            global_step += 1
            tokens_seen += input_batch.numel()

            # 7b) Optional evaluation step
            if global_step % config.eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    config.model, config.train_loader, config.val_loader, config.device, config.eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_loader_accuracy(config.train_loader, config.model, config.device)
        val_accuracy = calc_loader_accuracy(config.val_loader, config.model, config.device)
        print(f"Training accuracy: {train_accuracy*100:.3f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.3f}%")
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    return TrainingResults(
        train_losses=train_losses,
        val_losses=val_losses,
        track_tokens_seen=track_tokens_seen, # Pass steps seen
        train_accuracies=train_accuracies,   # Pass train accuracy
        val_accuracies=val_accuracies     # Pass validation accuracy
        # track_lrs is not handled by train_model_simple
    )

def train_model(config: TrainingConfig):
    # Added lists for accuracy
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    train_accuracies, val_accuracies = [], [] # Added accuracy lists
    tokens_seen, global_step = 0, -1

    peak_lr = config.optimizer.param_groups[0]["lr"]
    total_training_steps = len(config.train_loader) * config.num_epochs
    # Prevent division by zero if warmup_steps is 0 or None
    warmup_steps = config.warmup_steps or 0
    lr_increment = (peak_lr - config.initial_lr) / warmup_steps if warmup_steps > 0 else 0

    for epoch in range(config.num_epochs):
        config.model.train()
        for input_batch, target_batch in config.train_loader:
            config.optimizer.zero_grad()
            global_step += 1

            # Learning rate scheduling
            if warmup_steps > 0 and global_step < warmup_steps:
                lr = config.initial_lr + global_step * lr_increment
            else:
                # Cosine annealing after warmup
                progress = ((global_step - warmup_steps) / 
                            (total_training_steps - warmup_steps))
                lr = config.min_lr + (peak_lr - config.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            for param_group in config.optimizer.param_groups:
                param_group["lr"] = lr
            # Only track if schedule is active
            if warmup_steps > 0 or total_training_steps > 0:
                track_lrs.append(lr)

            # Calculate loss using updated function (derives classification logit)
            loss = calc_batch_loss(input_batch, target_batch, config.model, config.device)
            loss.backward()

            # Gradient clipping (optional but often helpful)
            if global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(config.model.parameters(), max_norm=1.0)

            config.optimizer.step()
            tokens_seen += input_batch.numel()
            
            # Evaluation step
            if global_step % config.eval_freq == 0:
                # Unpack all four values returned by evaluate_model
                train_loss, val_loss = evaluate_model(
                    config.model, config.train_loader, config.val_loader,
                    config.device, config.eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # Print the current losses
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        # Calculate accuracy after each epoch
        train_accuracy = calc_loader_accuracy(config.train_loader, config.model, config.device, num_batches=config.eval_iter)
        val_accuracy = calc_loader_accuracy(config.val_loader, config.model, config.device, num_batches=config.eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.3f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.3f}%")
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)


    return TrainingResults(
        train_losses=train_losses,
        val_losses=val_losses,
        track_tokens_seen=track_tokens_seen, # Pass steps seen
        train_accuracies=train_accuracies,   # Pass train accuracy
        val_accuracies=val_accuracies,     # Pass validation accuracy
        track_lrs=track_lrs if (warmup_steps > 0 or total_training_steps > 0) else None
    )

