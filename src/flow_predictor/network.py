import torch
from sklearn.metrics import r2_score
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
            input_size, hidden_size, num_layers, dropout=0.2, batch_first=True
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
    batch_size: int = 256,
    verbose: bool = True,
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size, shuffle=True)
    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = net(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                train_loss = total_loss / len(loader)
                test_pred = net(X_test)
                test_loss = criterion(test_pred, y_test).item()
                test_r2 = r2_score(y_test.cpu().numpy(), test_pred.cpu().numpy())
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Test Lost: {test_loss:.4f}, Test R2: {test_r2:.4f}"
                )
            net.train()
    return net
