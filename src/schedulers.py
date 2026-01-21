"""Schedulers for RVNP training.

This module provides:
1. Learning rate scheduling (patience-based, like PyTorch's ReduceLROnPlateau)
2. Sample scheduling (gradual increase of posterior/simulator samples during training)

Since Optax doesn't natively support validation-based scheduling, we implement stateful
schedulers that track metrics and trigger changes.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LRSchedulerState:
    """Tracks learning rate reduction state.

    Attributes:
        current_lr: Current learning rate
        best_val_loss: Best validation loss seen so far
        epochs_since_improvement: Number of epochs without improvement
        num_reductions: Total number of times LR has been reduced
    """
    current_lr: float
    best_val_loss: float
    epochs_since_improvement: int
    num_reductions: int


class ReduceLROnPlateau:
    """Patience-based learning rate reducer for JAX/Optax.

    Reduces learning rate when validation loss plateaus. This scheduler is designed
    to work with JAX's functional optimizer paradigm by maintaining explicit state
    and signaling when the optimizer needs to be reconstructed with a new learning rate.

    Example usage:
        ```python
        # Initialize scheduler
        scheduler = ReduceLROnPlateau(
            initial_lr=0.1,
            patience=20,
            factor=0.1,
            min_lr=1e-4
        )
        state = scheduler.init_state()

        # In training loop
        for epoch in range(max_epochs):
            # ... training ...

            # Update scheduler with validation loss
            state, lr_changed = scheduler.step(state, val_loss)

            # Reconstruct optimizer if LR changed
            if lr_changed:
                optimizer = get_optimizer(config, lr_override=state.current_lr)
                opt_state = optimizer.init(params)
        ```

    Args:
        initial_lr: Initial learning rate (e.g., 0.1)
        patience: Number of epochs to wait before reducing LR (e.g., 20)
        factor: Factor by which to reduce LR (e.g., 0.1 means divide by 10)
        min_lr: Minimum learning rate threshold (e.g., 1e-4)
        verbose: Whether to print LR reduction messages
    """

    def __init__(
        self,
        initial_lr: float,
        patience: int,
        factor: float,
        min_lr: float,
        verbose: bool = True
    ):
        """Initialize the scheduler.

        Args:
            initial_lr: Starting learning rate
            patience: Epochs to wait before reducing LR
            factor: Multiplicative factor for LR reduction (< 1.0)
            min_lr: Minimum LR (scheduler stops reducing below this)
            verbose: Print messages when LR is reduced
        """
        self.initial_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose

        # Validation
        if factor >= 1.0:
            raise ValueError(f"factor must be < 1.0, got {factor}")
        if initial_lr < min_lr:
            raise ValueError(f"initial_lr ({initial_lr}) must be >= min_lr ({min_lr})")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

    def init_state(self) -> LRSchedulerState:
        """Initialize scheduler state.

        Returns:
            Initial scheduler state with current_lr set to initial_lr
        """
        return LRSchedulerState(
            current_lr=self.initial_lr,
            best_val_loss=float('inf'),
            epochs_since_improvement=0,
            num_reductions=0
        )

    def step(
        self,
        state: LRSchedulerState,
        val_loss: float
    ) -> Tuple[LRSchedulerState, bool]:
        """Update scheduler based on validation loss.

        This method should be called once per epoch with the current validation loss.
        It tracks the best validation loss and reduces the learning rate if no improvement
        is seen for `patience` consecutive epochs.

        Args:
            state: Current scheduler state
            val_loss: Validation loss for this epoch

        Returns:
            Tuple of (new_state, lr_changed):
                - new_state: Updated scheduler state
                - lr_changed: True if learning rate was reduced, False otherwise
        """
        lr_changed = False

        # Check if validation loss improved
        if val_loss < state.best_val_loss:
            # Improvement detected - reset patience counter
            new_state = LRSchedulerState(
                current_lr=state.current_lr,
                best_val_loss=val_loss,
                epochs_since_improvement=0,
                num_reductions=state.num_reductions
            )
        else:
            # No improvement - increment patience counter
            epochs_waiting = state.epochs_since_improvement + 1

            # Check if we should reduce learning rate
            if epochs_waiting >= self.patience:
                # Calculate new learning rate
                new_lr = max(state.current_lr * self.factor, self.min_lr)

                # Use tolerance to handle floating point precision
                # Only trigger reduction if LR decreases by at least 0.1% (relative)
                lr_reduction_threshold = state.current_lr * 0.001
                if (state.current_lr - new_lr) > lr_reduction_threshold:
                    # LR was actually reduced (not at minimum yet)
                    lr_changed = True

                    if self.verbose:
                        print(f"\n{'='*60}")
                        print(f"ReduceLROnPlateau: Reducing learning rate")
                        print(f"  {state.current_lr:.2e} → {new_lr:.2e}")
                        print(f"  Reason: No improvement for {self.patience} epochs")
                        print(f"  Best val loss: {state.best_val_loss:.6f}")
                        print(f"  Reduction #{state.num_reductions + 1}")
                        print(f"{'='*60}\n")

                    new_state = LRSchedulerState(
                        current_lr=new_lr,
                        best_val_loss=state.best_val_loss,  # Keep best so far
                        epochs_since_improvement=0,  # Reset counter after reduction
                        num_reductions=state.num_reductions + 1
                    )
                else:
                    # Already at minimum LR - no change
                    if self.verbose and epochs_waiting == self.patience:
                        print(f"\nReduceLROnPlateau: At minimum LR ({self.min_lr:.2e}), not reducing further\n")

                    new_state = LRSchedulerState(
                        current_lr=state.current_lr,
                        best_val_loss=state.best_val_loss,
                        epochs_since_improvement=epochs_waiting,
                        num_reductions=state.num_reductions
                    )
            else:
                # Still within patience window
                new_state = LRSchedulerState(
                    current_lr=state.current_lr,
                    best_val_loss=state.best_val_loss,
                    epochs_since_improvement=epochs_waiting,
                    num_reductions=state.num_reductions
                )

        return new_state, lr_changed

    def get_last_lr(self, state: LRSchedulerState) -> float:
        """Get the current learning rate from state.

        Args:
            state: Current scheduler state

        Returns:
            Current learning rate
        """
        return state.current_lr
