import pytest
import torch

from models.lstm_autoencoder import LSTMAutoencoder, load_config


def _make_model(input_size=6, hidden_size=8, latent_size=4, num_layers=1):
    return LSTMAutoencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        dropout=0.0,
        recurrent_dropout=0.0,
    )


def test_forward_pass_output_shape_matches_input():
    model = _make_model(input_size=6)
    model.eval()
    x = torch.randn(3, 10, 6)  # (batch, seq_len, features)

    with torch.no_grad():
        out = model(x)

    assert out.shape == x.shape


def test_forward_pass_handles_batch_size_one_and_multi_layer():
    model = _make_model(input_size=5, num_layers=2)
    model.eval()
    x = torch.randn(1, 20, 5)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 20, 5)


def test_forward_pass_output_is_finite():
    model = _make_model(input_size=4)
    model.eval()
    x = torch.randn(2, 15, 4)

    with torch.no_grad():
        out = model(x)

    assert torch.isfinite(out).all()


def test_from_config_builds_model_with_expected_dimensions(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  input_size: 6
  hidden_size: 8
  latent_size: 4
  num_layers: 1
  dropout: 0.0
  recurrent_dropout: 0.0
""",
        encoding="utf-8",
    )

    model = LSTMAutoencoder.from_config(config_path)

    assert isinstance(model, LSTMAutoencoder)
    assert model.encoder.input_size == 6
    assert model.encoder.hidden_size == 8
    assert model.encoder_fc.out_features == 4


def test_from_config_missing_keys_raises(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  input_size: 6
  hidden_size: 8
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        LSTMAutoencoder.from_config(config_path)


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")
