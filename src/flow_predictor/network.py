import lightning as L
from torch import nn, optim


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


class LtModule(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        meta: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss = self.criterion(self.net(X), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        loss = self.criterion(self.net(X), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y = batch
        loss = self.criterion(self.net(X), y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return optim.Adam(
            self.net.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
