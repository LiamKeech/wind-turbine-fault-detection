from __future__ import annotations
from typing import Literal
from torch import nn
import torch

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for multivariate time series reconstruction.

    Args:
        nn (_type_): _PyTorch module base class_.
    """    

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the LSTMAutoencoder.

        Args:
            input_size (int): Number of features in the input data.
            hidden_size (int): Size of the hidden state in the LSTM layers.
            latent_size (int): Size of the latent representation.
            num_layers (int, optional): Number of LSTM layers for encoder and decoder. Defaults to 2.
            dropout (float, optional): Dropout probability for LSTM layers (applies when num_layers > 1). Defaults to 0.0.
        """        
        
        super().__init__()
        self._validate_hyperparams(input_size, hidden_size, latent_size, num_layers, dropout)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.hidden_to_latent = nn.Linear(hidden_size, latent_size)
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    @staticmethod
    def _validate_hyperparams(
        input_size: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """
        Validate hyperparameters for the LSTM autoencoder.

        Args:
            input_size (int): Number of features in the input data.
            hidden_size (int): Size of the hidden state in the LSTM layers.
            latent_size (int): Size of the latent representation.
            num_layers (int): Number of LSTM layers for encoder and decoder.
            dropout (float): Dropout probability for LSTM layers (applies when num_layers > 1).

        Raises:
            ValueError: If any hyperparameter is invalid.
        """              
        
        if input_size < 1:
            raise ValueError("input_size must be >= 1.")
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1.")
        if latent_size < 1:
            raise ValueError("latent_size must be >= 1.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        """
        Validate input tensor shape for the LSTM autoencoder.

        Raises:
            ValueError: If the input tensor has an invalid shape.
            ValueError: If the input feature dimension is invalid.
        """        
        
        if x.dim() != 3:
            raise ValueError("Input must have shape (batch, seq_len, features).")
        if x.size(-1) < 1:
            raise ValueError("Input feature dimension must be >= 1.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input sequence into the latent space.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, seq_len, features).

        Returns:
            torch.Tensor: Latent tensor with shape (batch, latent_size).
        """        
        
        self._validate_input(x)
        _, (h_n, _) = self.encoder(x)
        last_hidden = h_n[-1]
        return self.hidden_to_latent(last_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, seq_len, features).

        Returns:
            torch.Tensor: Reconstructed tensor with shape (batch, seq_len, features).
        """        
        
        self._validate_input(x)
        latent = self.encode(x)
        decoder_seed = self.latent_to_hidden(latent)
        decoder_input = decoder_seed.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(decoder_input)
        return self.output_layer(decoded)

    def reconstruction_error(
        self,
        x: torch.Tensor,
        *,
        reduction: Literal["sample", "none"] = "sample",
    ) -> torch.Tensor:
        """
        Compute the reconstruction error between the input and the autoencoder output.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, seq_len, features).
            reduction (Literal["sample", "none"], optional): Reduction method. Defaults to "sample".

        Raises:
            ValueError: If the reduction method is invalid.

        Returns:
            torch.Tensor: Reconstruction error tensor.
        """        
        
        reconstruction = self.forward(x)
        error = (reconstruction - x).pow(2)
        if reduction == "none":
            return error
        if reduction == "sample":
            return error.mean(dim=(1, 2))
        raise ValueError("reduction must be 'sample' or 'none'.")