from pathlib import Path
from typing import Any, Dict
from torch import nn
import torch
import yaml

def load_config(config_path: Path) -> Dict[str, Any]:
	"""
	Load YAML configuration file.

	Args:
		config_path (Path): Path to YAML configuration file.

	Returns:
		Dict[str, Any]: Configuration dictionary parsed from YAML file.
	"""
 
	with open(config_path, "r", encoding="utf-8") as file:
		return yaml.safe_load(file) or {}
class LSTMAutoencoder(nn.Module):
	def __init__(
		self,
		input_size: int,
		hidden_size: int,
		latent_size: int,
		num_layers: int,
		dropout: float,
		recurrent_dropout: float,
	) -> None:
		"""
		Initialize LSTM Autoencoder model.

		Args:
			input_size (int): Number of input features per time step.
			hidden_size (int): Number of hidden units in LSTM layers.
			latent_size (int): Size of latent representation.
			num_layers (int): Number of LSTM layers in encoder and decoder.
			dropout (float): Dropout rate for fully connected layers.
			recurrent_dropout (float): Dropout rate for recurrent connections.

		Returns:
			None
		"""
  
		super().__init__()
		self.num_layers = num_layers
		self.dropout = nn.Dropout(dropout)
		self.recurrent_dropout = float(recurrent_dropout)
		self.encoder = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout,
		)
		self.encoder_fc = nn.Linear(hidden_size, latent_size)
		self.decoder_fc = nn.Linear(latent_size, hidden_size)
		self.decoder = nn.LSTM(
			input_size=hidden_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout,
		)
		self.output_layer = nn.Linear(hidden_size, input_size)

	def _locked_dropout(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Apply locked/variational dropout to recurrent connections.

		Args:
			x (torch.Tensor): Input tensor to apply dropout to.

		Returns:
			torch.Tensor: Tensor with locked dropout applied.
		"""
  
		if not self.training or self.recurrent_dropout <= 0:
			return x
		mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.recurrent_dropout)
		mask = mask / (1 - self.recurrent_dropout)
		return x * mask

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass through the autoencoder.

		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

		Returns:
			torch.Tensor: Reconstructed tensor of same shape as input.
		"""
  
		x = self.dropout(x)
		x = self._locked_dropout(x)
		_, (hidden, _) = self.encoder(x)
		latent = self.encoder_fc(hidden[-1])
		latent = self.dropout(latent)
		hidden_state = torch.tanh(self.decoder_fc(latent))
		seq_len = x.size(1)
		dec_input = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
		dec_input = self._locked_dropout(dec_input)
		dec_h0 = hidden_state.unsqueeze(0).repeat(self.num_layers, 1, 1)
		dec_c0 = torch.zeros_like(dec_h0)
		dec_out, _ = self.decoder(dec_input, (dec_h0, dec_c0))
		dec_out = self.dropout(dec_out)
		return self.output_layer(dec_out)

	@classmethod
	def from_config(cls, config_path: Path) -> "LSTMAutoencoder":
		"""
		Instantiate LSTMAutoencoder from configuration file.

		Args:
			config_path (Path): Path to YAML configuration file containing model parameters.

		Returns:
			LSTMAutoencoder: Initialized model instance.
		"""
  
		config = load_config(config_path)
		model_cfg = config.get("model", {})
		required = [
			"input_size",
			"hidden_size",
			"latent_size",
			"num_layers",
			"dropout",
			"recurrent_dropout",
		]
		missing = [key for key in required if key not in model_cfg]
		if missing:
			raise ValueError(f"Missing model config keys: {missing}")
		return cls(
			input_size=int(model_cfg["input_size"]),
			hidden_size=int(model_cfg["hidden_size"]),
			latent_size=int(model_cfg["latent_size"]),
			num_layers=int(model_cfg["num_layers"]),
			dropout=float(model_cfg["dropout"]),
			recurrent_dropout=float(model_cfg["recurrent_dropout"]),
		)