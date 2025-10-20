"""PyTorch-based LSTM classifier for candle direction forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mlops.data_loader import load_parquet
from mlops.evaluation import classification_metrics
from mlops.features import FEATURE_COLUMNS, make_basic_features, make_label
from mlops.split import time_train_test_split

FEATURE_ORDER = FEATURE_COLUMNS


@dataclass
class LSTMTrainingConfig:
    sequence_length: int = 60
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-3


class _SequenceDataset(Dataset):
    """Torch dataset wrapping feature sequences and binary labels."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        if sequences.ndim != 3:
            raise ValueError("Expected sequences with shape (n, seq_len, n_features)")
        if len(sequences) != len(labels):
            raise ValueError("Sequences and labels must have matching lengths")
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:  # pragma: no cover - simple proxy
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class _LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.output(last)
        return logits.squeeze(-1)


def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame of features")
    return X


def _ensure_series(y: pd.Series | Sequence) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    return pd.Series(y)


def _build_sequences(
    features: pd.DataFrame,
    sequence_length: int,
) -> Tuple[np.ndarray, List[pd.Index]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if features.empty:
        return np.empty((0, sequence_length, len(features.columns))), []

    ordered = features.sort_index()
    values = ordered.to_numpy(dtype=np.float32)
    if len(values) < sequence_length:
        return np.empty((0, sequence_length, values.shape[1])), []

    seqs: List[np.ndarray] = []
    indices: List[pd.Index] = []
    for end_pos in range(sequence_length - 1, len(values)):
        seq = values[end_pos - sequence_length + 1 : end_pos + 1]
        seqs.append(seq)
        indices.append(ordered.index[end_pos])

    return np.stack(seqs), indices


class LSTMForecaster:
    """Wrapper providing a scikit-like interface around a PyTorch LSTM."""

    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        feature_order: Iterable[str] | None = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.feature_order = list(feature_order or FEATURE_ORDER)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.model = _LSTMClassifier(
            input_size=len(self.feature_order),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

    def _prepare_sequences(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[pd.Index]]:
        df = _ensure_dataframe(X).loc[:, self.feature_order]
        scaled = self.scaler.transform(df)
        scaled_df = pd.DataFrame(scaled, index=df.index, columns=self.feature_order)
        return _build_sequences(scaled_df, self.sequence_length)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        epochs: int = 15,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
    ) -> None:
        X_df = _ensure_dataframe(X_train).loc[:, self.feature_order]
        y_series = _ensure_series(y_train)

        self.scaler.fit(X_df)
        sequences, indices = self._prepare_sequences(X_df)
        if not len(indices):
            raise RuntimeError("Insufficient data to build LSTM sequences")
        y_values = y_series.loc[indices].to_numpy(dtype=np.float32)

        dataset = _SequenceDataset(sequences, y_values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for _ in range(epochs):  # pragma: no cover - training loop difficult to unit test
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimiser.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimiser.step()

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        sequences, indices = self._prepare_sequences(X)
        if not len(indices):
            return pd.Series(dtype="float64")

        tensor = torch.tensor(sequences, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():  # pragma: no branch - inference
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return pd.Series(probs.astype(np.float64), index=indices)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        probabilities = self.predict_proba(X)
        if probabilities.empty:
            return pd.Series(dtype="int64")
        return (probabilities >= threshold).astype("int64")

    def predict_latest_proba(self, X: pd.DataFrame) -> float:
        probabilities = self.predict_proba(X)
        if probabilities.empty:
            raise ValueError("Insufficient history for LSTM prediction")
        return float(probabilities.iloc[-1])

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "scaler": {
                "mean_": self.scaler.mean_.tolist(),
                "scale_": self.scaler.scale_.tolist(),
            },
            "sequence_length": self.sequence_length,
            "feature_order": self.feature_order,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
        torch.save(payload, file_path)

    @classmethod
    def load(cls, path: str | Path, map_location: Optional[str] = None) -> "LSTMForecaster":
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        payload = torch.load(file_path, map_location=map_location)
        scaler_state = payload.get("scaler", {})
        model = cls(
            sequence_length=int(payload.get("sequence_length", 60)),
            hidden_size=int(payload.get("hidden_size", 64)),
            num_layers=int(payload.get("num_layers", 2)),
            dropout=float(payload.get("dropout", 0.1)),
            feature_order=payload.get("feature_order", FEATURE_ORDER),
            device=torch.device(map_location) if map_location else None,
        )
        state_dict = payload.get("state_dict")
        if state_dict is None:
            raise ValueError("Serialized model missing state_dict")
        model.model.load_state_dict(state_dict)
        if scaler_state:
            model.scaler.mean_ = np.asarray(scaler_state.get("mean_", []), dtype=np.float64)
            model.scaler.scale_ = np.asarray(scaler_state.get("scale_", []), dtype=np.float64)
        else:
            raise ValueError("Serialized model missing scaler state")
        return model


def train_lstm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Optional[LSTMTrainingConfig] = None,
) -> LSTMForecaster:
    cfg = config or LSTMTrainingConfig()
    model = LSTMForecaster(
        sequence_length=cfg.sequence_length,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        feature_order=FEATURE_ORDER,
    )
    model.fit(
        X_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
    )
    return model


def predict_lstm(model: LSTMForecaster, X: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(X)
    return predictions.to_numpy(dtype=np.int64)


def predict_lstm_proba(model: LSTMForecaster, X: pd.DataFrame) -> pd.Series:
    return model.predict_proba(X)


def save_model(model: LSTMForecaster, path: str | Path) -> None:
    model.save(path)


def load_model(path: str | Path) -> LSTMForecaster:
    return LSTMForecaster.load(path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM classifier.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/btc_usdt_1m_all.parquet"),
        help="Path to the Parquet dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/artifacts/lstm.pt"),
        help="Where to store the trained model artifact.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Label horizon used for training and evaluation.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Number of timesteps in each LSTM input sequence.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    df = load_parquet(args.data)
    features = make_basic_features(df)
    labels = make_label(df, horizon=args.horizon)

    dataset = pd.concat(
        [
            features,
            labels.rename("label"),
            df[["timestamp", "datetime", "close"]],
        ],
        axis=1,
    ).dropna()

    sequence_length = max(int(args.sequence_length), 1)
    if len(dataset) <= sequence_length:
        raise RuntimeError("Not enough data to train the LSTM model")

    dataset = dataset.iloc[sequence_length - 1 :]
    X = dataset.loc[:, FEATURE_ORDER]
    y = dataset.loc[:, "label"].astype(int)

    X_train, X_test, y_train, y_test = time_train_test_split(X, y)

    config = LSTMTrainingConfig(
        sequence_length=sequence_length,
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
    )

    model = train_lstm(X_train, y_train, config=config)
    save_model(model, args.model_path)

    y_pred_series = model.predict(X_test)
    if y_pred_series.empty:
        print("Warning: no predictions produced for the evaluation split.")
        metrics = {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    else:
        y_true = y_test.loc[y_pred_series.index]
        metrics = classification_metrics(y_true, y_pred_series)

    train_last = dataset.loc[X_train.index, "datetime"].iloc[-1] if not X_train.empty else None
    test_last = dataset.loc[y_pred_series.index, "datetime"].iloc[-1] if not y_pred_series.empty else None

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    if train_last is not None:
        print(f"Train last timestamp: {pd.to_datetime(train_last).isoformat()}")
    if test_last is not None:
        print(f"Test last timestamp: {pd.to_datetime(test_last).isoformat()}")

    print("Classification metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print(f"Model saved to {args.model_path.resolve()}")


if __name__ == "__main__":
    main()
