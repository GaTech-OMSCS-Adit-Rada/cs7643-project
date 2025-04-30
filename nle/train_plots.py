"""Contains functions to plot training and validation metrics over epochs and tokens seen."""

import torch
from nle.train import TrainingConfig, TrainingResults
from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def _plot_metric_vs_epochs(
    training_config: TrainingConfig,
    tokens_seen: List[float], # Or torch.Tensor if you keep them as tensors
    train_data: List[float], # Or torch.Tensor
    val_data: List[float], # Or torch.Tensor
    train_label: str,
    val_label: str,
    y_label: str,
    filepath: Optional[str] = None
):
    """
    Helper function to plot training and validation metrics against epochs and tokens seen.

    Args:
        training_config: Configuration object containing num_epochs.
        tokens_seen: List or Tensor of cumulative tokens seen at each evaluation point.
        train_data: List or Tensor of training metric values.
        val_data: List or Tensor of validation metric values.
        train_label: Label for the training data line.
        val_label: Label for the validation data line.
        y_label: Label for the primary y-axis.
        filepath: Optional path to save the figure.
    """
    if not train_data:
        print(f"Warning: No training data provided for metric '{y_label}'. Skipping plot.")
        return
    if not val_data:
        print(f"Warning: No validation data provided for metric '{y_label}'. Skipping plot.")
        return
    if len(train_data) != len(val_data):
         print(f"Warning: Training data ({len(train_data)} points) and validation data ({len(val_data)} points) for metric '{y_label}' have different lengths. Plotting might be misleading.")
         # Decide how to handle this - plot anyway, truncate, raise error? Plotting for now.
    if len(train_data) != len(tokens_seen):
         print(f"Warning: Metric data ({len(train_data)} points) and tokens_seen ({len(tokens_seen)} points) for metric '{y_label}' have different lengths. Token axis might be misaligned.")
         # Decide how to handle this. Plotting for now.


    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Ensure epochs_seen aligns with the *number* of data points recorded
    num_data_points = len(train_data)
    epochs_seen = torch.linspace(0, training_config.num_epochs, num_data_points)

    # Plot training and validation data against epochs
    ax1.plot(epochs_seen, train_data, label=train_label)
    # Ensure validation data is plotted against the same epoch scale
    # If lengths differ, this might slice val_data or cause error depending on versions/data
    ax1.plot(epochs_seen[:len(val_data)], val_data[:len(epochs_seen)], linestyle="-.", label=val_label)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(y_label)
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny() # Create a second x-axis that shares the same y-axis
    # The y-values here don't matter, only the x-values (tokens_seen) for axis alignment
    ax2.plot(tokens_seen, train_data, alpha=0) # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    # Optional: Format token ticks if they get large (e.g., K, M)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K' if x >= 1e3 else f'{x:.0f}'))

    # Highlight best
    min_val_loss = min(val_data)
    min_val_idx = val_data.index(min_val_loss) # Gets the first occurrence
    min_epoch = epochs_seen[min_val_idx]
    # 1. Highlight the point
    ax1.scatter(min_epoch, min_val_loss, color='red', s=50, zorder=5, label=f'Best {val_label}') # zorder makes it plot on top
    # 2. Annotate the point
    annotation_text = f'Val Min: {min_val_loss:.3f}' # Format as needed
    ax1.annotate(annotation_text,
                    xy=(min_epoch, min_val_loss), # Point to annotate
                    xytext=(min_epoch-0.1, min_val_loss + (max(val_data) - min_val_loss)*0.3), # Position of text (adjust offset as needed)
                    textcoords='data',
                    ha='center', # Horizontal alignment
                    va='bottom', # Vertical alignment
                    arrowprops=dict(arrowstyle="->", color='red', connectionstyle="arc3,rad=.2"))


    fig.tight_layout() # Adjust layout to make room
    if filepath:
        try:
            plt.savefig(filepath)
            print(f"Plot saved to {filepath}")
        except Exception as e:
            print(f"Error saving plot to {filepath}: {e}")
    plt.grid()
    plt.show()


def plot_losses(training_config: TrainingConfig, training_results: TrainingResults, filepath=None):
    """Plots training and validation losses against epochs and tokens seen."""
    print("Plotting losses...")
    _plot_metric_vs_epochs(
        training_config=training_config,
        # Using track_tokens_seen as in the original function
        tokens_seen=training_results.track_tokens_seen,
        train_data=training_results.train_losses,
        val_data=training_results.val_losses,
        train_label="Training loss",
        val_label="Validation loss",
        y_label="Loss",
        filepath=filepath
    )


def plot_perplexity(training_config: TrainingConfig, training_results: TrainingResults, filepath=None):
    """Plots training and validation perplexity against epochs and tokens seen."""
    print("Plotting perplexity...")
    _plot_metric_vs_epochs(
        training_config=training_config,
        # Using track_tokens as in the original function
        tokens_seen=training_results.track_tokens_seen,
        train_data=training_results.train_perplexity,
        val_data=training_results.val_perplexity,
        train_label="Training perplexity",
        val_label="Validation perplexity",
        y_label="Perplexity",
        filepath=filepath
    )

def plot_and_save_learning_rate(training_results: TrainingResults, filepath=None):
    plt.figure(figsize=(7, 5))
    plt.plot(range(len(training_results.track_lrs)), training_results.track_lrs)
    plt.title("Learning Rate Schedule")
    plt.ylabel("Learning rate")
    plt.xlabel("Steps")
    plt.grid()
    # Format y-axis to scientific notation
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    if filepath:
        plt.savefig(filepath)
    plt.show()
