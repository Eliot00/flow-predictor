from collections.abc import Sequence

import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class LSTMModel(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, output_size: int
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=0.2 if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last_out)  # (batch, output_size)


def train_torch_model(
    net: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    grad_clip: float | None = 1.0,
    target_scaler: StandardScaler | None = None,
    target_names: Sequence[str] | None = None,
    verbose: bool = True,
) -> nn.Module:
    """Train a torch regressor.

    ``y_train`` and ``y_test`` may be in a transformed space.  If
    ``target_scaler`` is provided, losses are optimized in that space while
    the reported metrics are converted back to the original target units.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = net(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                train_loss = total_loss / len(loader)
                test_pred = net(X_test)
                test_loss = criterion(test_pred, y_test).item()

                y_true_np = y_test.cpu().numpy()
                y_pred_np = test_pred.cpu().numpy()
                if target_scaler is not None:
                    y_true_np = target_scaler.inverse_transform(y_true_np)
                    y_pred_np = target_scaler.inverse_transform(y_pred_np)

                target_mse = mean_squared_error(
                    y_true_np, y_pred_np, multioutput="raw_values"
                )
                target_r2 = r2_score(y_true_np, y_pred_np, multioutput="raw_values")
                names = target_names or [f"target_{i}" for i in range(len(target_r2))]
                details = " | ".join(
                    f"{name}: MSE={mse:.2f}, R²={r2:.3f}"
                    for name, mse, r2 in zip(names, target_mse, target_r2)
                )
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss (scaled): {train_loss:.4f}, "
                    f"Test Loss (scaled): {test_loss:.4f}, "
                    f"Test R² (mean): {target_r2.mean():.4f}\n  {details}"
                )
            net.train()
    return net
