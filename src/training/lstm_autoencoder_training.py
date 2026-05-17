from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def resolve_device(device_value: str) -> torch.device:
	"""
	Resolve device string to torch.device object.

	Args:
		device_value (str): Device specification string ('auto', 'cpu', 'cuda', etc.).

	Returns:
		torch.device: Resolved device object. If 'auto', returns GPU if available, otherwise CPU.
	"""
 
	if device_value == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device_value)

def prepare_dataloaders(
	train_sequences,
	val_sequences,
	batch_size: int,
	train_labels: Optional[np.ndarray] = None,
	normal_label: int = 0,
) -> Tuple[DataLoader, DataLoader]:
	"""
	Prepare training and validation dataloaders.

	Args:
		train_sequences: Array of training sequences.
		val_sequences: Array of validation sequences.
		batch_size (int): Batch size for dataloaders.
		train_labels (Optional[np.ndarray]): Labels for training sequences. If provided, only normal sequences are used. Default is None.
		normal_label (int): Label value indicating normal samples. Default is 0.

	Returns:
		Tuple[DataLoader, DataLoader]: Training and validation dataloaders.
	"""
 
	if train_labels is not None:
		labels = np.asarray(train_labels).astype(int)
		if labels.shape[0] != len(train_sequences):
			raise ValueError("train_labels length must match train_sequences length.")
		mask = labels == normal_label
		if not np.any(mask):
			raise ValueError("No normal sequences available for training.")
		train_sequences = train_sequences[mask]

	train_ds = TensorDataset(torch.tensor(train_sequences, dtype=torch.float32))
	val_ds = TensorDataset(torch.tensor(val_sequences, dtype=torch.float32))
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
	return train_loader, val_loader

def train_lstm_autoencoder(
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	training_cfg: Dict,
	device: torch.device,
) -> Dict[str, list]:
	"""
	Train LSTM autoencoder with early stopping.

	Args:
		model (nn.Module): LSTM autoencoder model to train.
		train_loader (DataLoader): Training dataloader.
		val_loader (DataLoader): Validation dataloader.
		training_cfg (Dict): Training configuration dictionary with keys: epochs, learning_rate, weight_decay, grad_clip, early_stopping_patience, early_stopping_min_delta, random_seed.
		device (torch.device): Device to train on.

	Returns:
		Dict[str, list]: Training history dictionary with train_loss, val_loss, best_val_loss, and stopped_epoch.
	"""
 
	torch.manual_seed(int(training_cfg.get("random_seed", 42)))
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=float(training_cfg["learning_rate"]),
		weight_decay=float(training_cfg.get("weight_decay", 0.0)),
	)
	max_grad_norm = float(training_cfg.get("grad_clip", 0.0))
	patience = int(training_cfg.get("early_stopping_patience", 0))
	min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
	best_val_loss = float("inf")
	best_state = None
	patience_counter = 0

	history = {"train_loss": [], "val_loss": []}
	epochs = int(training_cfg["epochs"])
	for epoch in range(epochs):
		model.train()
		train_loss = 0.0
		for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
			batch = batch.to(device)
			optimizer.zero_grad()
			recon = model(batch)
			loss = criterion(recon, batch)
			loss.backward()
			if max_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
			optimizer.step()
			train_loss += loss.item() * batch.size(0)
		train_loss /= len(train_loader.dataset)
		history["train_loss"].append(train_loss)

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for (batch,) in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
				batch = batch.to(device)
				recon = model(batch)
				loss = criterion(recon, batch)
				val_loss += loss.item() * batch.size(0)
		val_loss /= len(val_loader.dataset)
		history["val_loss"].append(val_loss)

		if val_loss < (best_val_loss - min_delta):
			best_val_loss = val_loss
			best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
			patience_counter = 0
			print(f"✓ Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} (Best)")
		else:
			patience_counter += 1
			print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
			if patience > 0 and patience_counter >= patience:
				print(f"Early stopping triggered at epoch {epoch+1}")
				break

	if best_state is not None:
		model.load_state_dict(best_state)

	history["best_val_loss"] = best_val_loss
	history["stopped_epoch"] = epoch + 1

	print(f"\n{'='*60}")
	print(f"Training Complete!")
	print(f"  Total Epochs: {epoch + 1}")
	print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
	print(f"  Final Val Loss: {history['val_loss'][-1]:.6f}")
	print(f"  Best Val Loss: {best_val_loss:.6f}")
	print(f"{'='*60}\n")

	return history

def save_model(model: nn.Module, output_path: Path) -> Path:
	"""
	Save trained model state dictionary to disk.

	Args:
		model (nn.Module): Trained model to save.
		output_path (Path): Destination file path for model weights.

	Returns:
		Path: Path where the model was saved.
	"""
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), output_path)
	return output_path

def plot_training_history(history: Dict[str, list]):
	"""
	Plot training and validation loss history.

	Args:
		history (Dict[str, list]): Training history dictionary with 'train_loss' and 'val_loss' keys.

	Returns:
		matplotlib.axes.Axes: Axes object with plotted training history.
	"""
 
	fig, ax = plt.subplots(figsize=(8, 3))
	ax.plot(history.get("train_loss", []), label="train")
	ax.plot(history.get("val_loss", []), label="val")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("MSE loss")
	ax.legend()
	return ax