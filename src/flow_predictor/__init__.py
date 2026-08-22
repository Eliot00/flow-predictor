from flow_predictor.network import MLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from flow_predictor.prepare import prepare_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Annotated
import pandas as pd
import typer
from flow_predictor.fake import fake_all

app = typer.Typer()

@app.command()
def cli(
    fake: Annotated[bool, typer.Option(help="Generate fake data.")] = False,
    train: Annotated[bool, typer.Option(help="Start training.")] = False,
    model: Annotated[str, typer.Option(help="Model type: 'linear' or 'mlp'")] = "linear",
):
    if fake:
        df = fake_all()
        print(f"Fake data: {df.describe()}")
    if train:
        df = fake_all()

        X_train_scaled, X_test_scaled, y_train, y_test, features = prepare_data(df, ["passby_visit", "entering_people", "dwell_people", "served_people"])

        if model == "linear":
            model_ = LinearRegression()
            model_.fit(X_train_scaled, y_train)

            y_pred = model_.predict(X_test_scaled)

            for i, col in enumerate(y_train.columns):
                mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
                r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
                print(f"{col}: MSE={mse:.2f}, R²={r2:.2f}")

            coef_df = pd.DataFrame(
                model_.coef_,
                index=y_train.columns,
                columns=features
            )
            print("\n系数矩阵（行=目标，列=特征）:")
            print(coef_df)
        elif model == "mlp":
            X_train_t = torch.tensor(X_train_scaled.values, dtype=torch.float32)
            y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
            X_test_t = torch.tensor(X_test_scaled.values, dtype=torch.float32)
            y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

            input_dim = X_train_t.shape[1]
            output_dim = y_train_t.shape[1]

            net = MLP(input_dim, output_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            batch_size = 256
            dataset = TensorDataset(X_train_t, y_train_t)
            loader = DataLoader(dataset, batch_size, shuffle=True)

            epochs = 50
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
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(loader):.4f}")

            net.eval()
            with torch.no_grad():
                y_pred_t = net(X_test_t)
                y_pred_np = y_pred_t.numpy()
                y_test_np = y_test_t.numpy()

                for i, col in enumerate(y_train.columns):
                    mse = mean_squared_error(y_test_np[:, i], y_pred_np[:, i])
                    r2 = r2_score(y_test_np[:, i], y_pred_np[:, i])
                    print(f"{col}: MSE={mse:.2f} R2={r2:.2f}")
        else:
            typer.echo("Unknown model.")

def main() -> None:
    app()
