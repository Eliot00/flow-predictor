from typing import Annotated

import pandas as pd
import torch
import typer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from flow_predictor.fake import fake_all
from flow_predictor.network import MLP, LSTMModel, train_torch_model
from flow_predictor.prepare import prepare_data, prepare_lstm_data
from flow_predictor.utils import set_seed

app = typer.Typer()


@app.command()
def cli(
    fake: Annotated[bool, typer.Option(help="Generate fake data.")] = False,
    train: Annotated[bool, typer.Option(help="Start training.")] = False,
    model: Annotated[
        str, typer.Option(help="Model type: 'linear', 'mlp' or 'lstm'")
    ] = "linear",
):
    set_seed()

    if fake:
        df = fake_all()
        print(f"Fake data: {df.describe()}")
    if train:
        df = fake_all()

        X_train_scaled, X_test_scaled, y_train, y_test, features = prepare_data(
            df, ["passby_visit", "entering_people", "dwell_people", "served_people"]
        )

        if model == "linear":
            model_ = LinearRegression()
            model_.fit(X_train_scaled, y_train)

            y_pred = model_.predict(X_test_scaled)

            for i, col in enumerate(y_train.columns):
                mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
                r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
                print(f"{col}: MSE={mse:.2f}, R²={r2:.2f}")

            coef_df = pd.DataFrame(
                model_.coef_, index=y_train.columns, columns=features
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
            train_torch_model(net, X_train_t, y_train_t, X_test_t, y_test_t)
        elif model == "lstm":
            X_train, y_train, X_test, y_test = prepare_lstm_data(
                df,
                target_cols=[
                    "passby_visit",
                    "entering_people",
                    "dwell_people",
                    "served_people",
                ],
            )
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32)
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            y_test_t = torch.tensor(y_test, dtype=torch.float32)

            input_size = X_train.shape[2]
            output_size = y_train.shape[1]
            net = LSTMModel(
                input_size, hidden_size=128, num_layers=2, output_size=output_size
            )
            train_torch_model(net, X_train_t, y_train_t, X_test_t, y_test_t)
        else:
            typer.echo("Unknown model.")


def main() -> None:
    app()
