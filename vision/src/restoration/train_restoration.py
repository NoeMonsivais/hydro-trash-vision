"""
Training loop for the simple CNN encoder-decoder restoration baseline.

Runs in two modes via the SMOKE_TEST flag below:
- SMOKE_TEST = True: tiny data subset, few epochs. Verifies the full
  train/validate/checkpoint pipeline runs end-to-end without crashing.
  Safe and fast on a laptop with no dedicated GPU.
- SMOKE_TEST = False: real training run, meant for cloud GPU (Colab/AWS/
  Azure), full dataset, enough epochs for a reportable result.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from deep_model import RestorationCNN
from dataset import UnderwaterRestorationDataset

SMOKE_TEST = True  # set to False for real training run

BATCH_SIZE = 4 if SMOKE_TEST else 16
EPOCHS = 2 if SMOKE_TEST else 50
LEARNING_RATE = 1e-4
SMOKE_TEST_SUBSET_SIZE = 20  # number of samples to use in smoke test

VISION_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = VISION_ROOT / "checkpoints" / "restoration_cnn"

def get_device() -> torch.device:
    """Pick CUDA if available, otherwise fall back to CPU.

    Written to be hardware-agnostic on purpose: the same script runs
    correctly (just much slower) on a laptop with no dedicated GPU and on
    a cloud GPU instance, with no code changes needed.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def run_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        train: bool,
) -> float:
    """Run one pass over the dataloader, either training or validating.

    The `train` flag fuses two identical loops into one. It controls whether 
    gradients are computed and weights updated (True), or if the model is 
    only evaluated (False) to save VRAM and time.
    """
    model.train() if train else model.eval()

    total_loss = 0.0

    # We use a context manager to dynamically turn off PyTorch's gradient 
    # engine during validation, preventing memory leaks.
    context = torch.enable_grad() if train else torch.no_grad()


    with context:
        for degraded, reference in dataloader:
            degraded = degraded.to(device)
            reference = reference.to(device)

            if train:
                optimizer.zero_grad()

            restored = model(degraded)
            loss = criterion(restored, reference)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
    

def main() -> None:
    """Orchestrate dataset loading, model initialization, and the training loop.
    
    Serves as the main entry point. It sets up the environment dynamically 
    based on the SMOKE_TEST flag to allow rapid local debugging before 
    committing to a long cloud run.
    """
    device = get_device()
    print(f"Using device: {device}")

    if SMOKE_TEST:
        print("Running in SMOKE TEST mode: tiny subset, few epochs.")

    train_dataset = UnderwaterRestorationDataset(split="train")
    val_dataset = UnderwaterRestorationDataset(split="val")

    if SMOKE_TEST:

        train_dataset = Subset(train_dataset, range(SMOKE_TEST_SUBSET_SIZE))
        val_dataset = Subset(val_dataset, range(min(SMOKE_TEST_SUBSET_SIZE, len(val_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RestorationCNN().to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.4f} - "
              f"val_loss: {val_loss:.4f} - {elapsed:.1f}s")

        # Strictly checkpoint based on validation performance to prevent 
        # saving an overfitted model that merely memorized the training data.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
            
            # Saving state_dict() (just the weights) is safer, lighter, and 
            # more portable than saving the entire model object.
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best val_loss, checkpoint saved to {checkpoint_path}")

    print(f"Training finished. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()