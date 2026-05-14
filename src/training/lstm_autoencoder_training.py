from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def resolve_device(device_value: str) -> torch.device:
	if device_value == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device_value)


def prepare_dataloaders(
	train_sequences,
	val_sequences,
	batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
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
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), output_path)
	return output_path


def plot_training_history(history: Dict[str, list]):
	fig, ax = plt.subplots(figsize=(8, 3))
	ax.plot(history.get("train_loss", []), label="train")
	ax.plot(history.get("val_loss", []), label="val")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("MSE loss")
	ax.legend()
	return ax
