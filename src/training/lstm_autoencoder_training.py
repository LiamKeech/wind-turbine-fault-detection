from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from features.lstm_autoencoder_features import WindowedSplits
from models.lstm_autoencoder import LSTMAutoencoder
import numpy as np
import torch

@dataclass(frozen=True)
class AutoencoderLoaders:
    """
    DataLoaders for LSTM autoencoder training.
    """    

    train: DataLoader
    val: DataLoader
    test: DataLoader

@dataclass(frozen=True)
class TrainingResult:
    """
    Result of LSTM autoencoder training, including history and best model state.
    """    

    history: Dict[str, List[float]]
    best_state_dict: Dict[str, torch.Tensor]
    best_val_loss: float

def get_device() -> torch.device:
    """
    Get the available torch device (GPU if available, otherwise CPU).

    Returns:
        torch.device: The device to use for training.
    """    
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataloaders(
    windowed: WindowedSplits,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
) -> AutoencoderLoaders:
    """
    Create DataLoaders for LSTM autoencoder training from windowed features.

    Args:
        windowed (WindowedSplits): Windowed features for train/validation/test splits.
        batch_size (int, optional): Batch size for each DataLoader. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the training loader. Defaults to True.

    Returns:
        AutoencoderLoaders: DataLoaders for train/val/test.
    """    
    
    train_loader = _make_loader(windowed.train.windows, batch_size, shuffle)
    val_loader = _make_loader(windowed.val.windows, batch_size, False)
    test_loader = _make_loader(windowed.test.windows, batch_size, False)
    return AutoencoderLoaders(train=train_loader, val=val_loader, test=test_loader)

def train_autoencoder(
    model: LSTMAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    device: Optional[torch.device] = None,
) -> TrainingResult:
    """
    Train the LSTM autoencoder with early stopping based on validation loss.

    Args:
        model (LSTMAutoencoder): The LSTM autoencoder to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        epochs (int, optional): Maximum number of training epochs. Defaults to 20.
        lr (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-3.
        patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 5.
        device (Optional[torch.device], optional): The device to use for training. Defaults to None.

    Returns:
        TrainingResult: The result of the training process.
    """    

    device = device or get_device()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")

    history: Dict[str, List[float]] = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_state_dict: Dict[str, torch.Tensor] = {}
    epochs_no_improve = 0

    for _ in range(epochs):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state_dict = _clone_state_dict(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state_dict:
        model.load_state_dict(best_state_dict)

    return TrainingResult(
        history=history,
        best_state_dict=best_state_dict,
        best_val_loss=best_val_loss,
    )

def _make_loader(windows: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Create a DataLoader from windowed features.

    Args:
        windows (np.ndarray): The windowed features.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: The created DataLoader.
    """    
    
    tensor = torch.tensor(windows, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def _run_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    """
    Run a single epoch of training or validation.

    Args:
        model (LSTMAutoencoder): The LSTM autoencoder to train or validate.
        loader (DataLoader): The DataLoader for the dataset.
        criterion (nn.Module): The loss function.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer for training.
        device (torch.device): The device to use for training or validation.

    Returns:
        float: The average loss for the epoch.
    """    
    
    model.train(optimizer is not None)
    running_loss = 0.0
    total = 0

    for (batch,) in loader:
        batch = batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        reconstruction = model(batch)
        loss = criterion(reconstruction, batch)
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * batch.size(0)
        total += batch.size(0)

    return running_loss / max(total, 1)

def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Create a deep copy of a model's state_dict with all tensors detached and moved to CPU.

    Args:
        state_dict (Dict[str, torch.Tensor]): The state dictionary to clone.

    Returns:
        Dict[str, torch.Tensor]: A deep copy of the state dictionary with all tensors detached and moved to CPU.
    """    
    
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}