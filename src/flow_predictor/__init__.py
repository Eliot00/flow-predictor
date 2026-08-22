from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import torch
import typer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from flow_predictor.fake import fake_all
from flow_predictor.network import MLP, LSTMModel, train_torch_model
from flow_predictor.prepare import prepare_data, prepare_lstm_data, scale_targets
from flow_predictor.utils import set_seed

app = typer.Typer()

TARGET_COLS = [
    "passby_visit",
    "entering_people",
    "dwell_people",
    "served_people",
]


@app.command()
def cli(
    data_file: Annotated[
        Path | None, typer.Option(help="Real data file path")
    ] = None,
    model: Annotated[
        str, typer.Option(help="Model type: 'linear', 'mlp' or 'lstm'")
    ] = "linear",
    epochs: Annotated[
        int, typer.Option(help="Number of training epochs (for MLP/LSTM)")
    ] = 50,
    batch_size: Annotated[
        int, typer.Option(help="Batch size for training (for MLP/LSTM)")
    ] = 32,
    hidden_size: Annotated[
        int, typer.Option(help="Hidden layer size for MLP / LSTM hidden units")
    ] = 64,
    lr: Annotated[
        float, typer.Option(help="Learning rate (for MLP/LSTM)")
    ] = 0.001,
):
    set_seed()
    if data_file is not None:
        if data_file.suffix == ".csv":
            df = pd.read_csv(data_file, parse_dates=["date"])
        elif data_file.suffix == "tsv":
            df = pd.read_csv(data_file, sep="\t", parse_dates=["date"])
        else:
            typer.echo("Unsupported file type")
            raise typer.Exit()
    else:
        df = fake_all()
        typer.echo("Using fake data")

    if model == "linear":
        X_train_scaled, X_test_scaled, y_train, y_test, features = prepare_data(
            df, ["passby_visit", "entering_people", "dwell_people", "served_people"]
        )
        model_ = LinearRegression()
        model_.fit(X_train_scaled, y_train)

        y_pred = model_.predict(X_test_scaled)

        for i, col in enumerate(y_train.columns):
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            print(f"{col}: MSE={mse:.2f}, R²={r2:.2f}")

        coef_df = pd.DataFrame(model_.coef_, index=y_train.columns, columns=features)
        print("\n系数矩阵（行=目标，列=特征）:")
        print(coef_df)
    elif model == "mlp":
        X_train_scaled, X_test_scaled, y_train, y_test, _ = prepare_data(
            df, TARGET_COLS
        )
        y_train_scaled, y_test_scaled, target_scaler = scale_targets(y_train, y_test)
        X_train_t = torch.tensor(X_train_scaled.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_scaled.values, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

        input_dim = X_train_t.shape[1]
        output_dim = y_train_t.shape[1]

        net = MLP(input_dim, output_dim, hidden_size=hidden_size)
        train_torch_model(
            net,
            X_train_t,
            y_train_t,
            X_test_t,
            y_test_t,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            target_scaler=target_scaler,
            target_names=TARGET_COLS,
        )
    elif model == "lstm":
        X_train, y_train, X_test, y_test = prepare_lstm_data(df, TARGET_COLS)
        y_train_scaled, y_test_scaled, target_scaler = scale_targets(y_train, y_test)
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

        input_size = X_train.shape[2]
        output_size = y_train_scaled.shape[1]
        net = LSTMModel(
            input_size, hidden_size=hidden_size, num_layers=2, output_size=output_size
        )
        train_torch_model(
            net,
            X_train_t,
            y_train_t,
            X_test_t,
            y_test_t,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            target_scaler=target_scaler,
            target_names=TARGET_COLS,
        )
    else:
        typer.echo("Unknown model.")


def main() -> None:
    app()
