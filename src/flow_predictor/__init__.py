import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
import numpy as np
import torch
import typer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from flow_predictor.fake import fake_all
from flow_predictor.network import MLP, LSTMModel, train_torch_model
from flow_predictor.prepare import (
    add_calendar_features,
    prepare_data,
    prepare_lstm_data,
    transform_targets,
)
from flow_predictor.utils import set_seed, get_weather

app = typer.Typer()

TARGET_COLS = [
    "passby_visit",
    "entering_people",
    "dwell_people",
    "served_people",
]

@app.command()
def prepare():
    # 临时模拟
    start_date = datetime(2026, 1, 1)
    end_date = datetime.now() - timedelta(days=1)

    current = start_date
    while current <= end_date:
        get_weather(lon=121.4455, lat=31.2264, date_str=current.strftime("%Y-%m-%d"))
        time.sleep(1)
        current += timedelta(days=1)

@app.command()
def train(
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
    lag: Annotated[
        str | None, typer.Option(help="Comma-separated lag days, e.g. '1,7'")
    ] = None,
    save_path: Annotated[
        Path | None, typer.Option(help="Directory or file path to save the trained model")
    ] = None
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
        X_train_scaled, X_test_scaled, y_train, y_test, features, _ = prepare_data(
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
        lag_days = [int(x.strip()) for x in lag.split(",")] if lag else None
        X_train_scaled, X_test_scaled, y_train, y_test, features, scaler = prepare_data(
            df, TARGET_COLS, lag_days
        )
        y_train_log, y_test_log = transform_targets(y_train.values, y_test.values)
        X_train_t = torch.tensor(X_train_scaled.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_log, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_scaled.values, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_log, dtype=torch.float32)

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
            target_names=TARGET_COLS,
        )
        if save_path:
            save_path = Path(save_path)
            if save_path.is_dir():
                save_path = save_path / f"mlp_model_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': net.state_dict(),
                'input_dim': input_dim,
                'hidden_size': hidden_size,
                'output_dim': output_dim,
                'feature_scaler': scaler,
                'feature_names': features,
                'target_names': TARGET_COLS,
                'model_type': 'mlp',
            }, save_path)
            typer.echo(f"Model saved to {save_path}")
    elif model == "lstm":
        X_train, y_train, X_test, y_test, features, scaler = prepare_lstm_data(df, TARGET_COLS)
        y_train_log, y_test_log = transform_targets(y_train, y_test)
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_log, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_log, dtype=torch.float32)

        input_size = X_train.shape[2]
        output_size = y_train_log.shape[1]
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
            target_names=TARGET_COLS,
        )
        if save_path:
            save_path = Path(save_path)
            if save_path.is_dir():
                save_path = save_path / f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # TODO: 标签数据需要一个统一的编码器
            le_category = LabelEncoder()
            le_category.fit(df['category'])

            torch.save({
                'model_state_dict': net.state_dict(),
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': 2,
                'output_size': output_size,
                'feature_scaler': scaler,
                'feature_names': features,
                'target_names': TARGET_COLS,
                'category_classes': sorted(df['category'].unique().tolist()),
                'model_type': 'lstm',
            }, save_path)
        typer.echo(f"LSTM model saved to {save_path}")
    else:
        typer.echo("Unknown model.")

@app.command()
def predict(model_file: Annotated[Path, typer.Option(help="Saved model file")], data_file: Annotated[Path, typer.Option(help="Stores, dates, features CSV file")]):
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    df = pd.read_csv(data_file, parse_dates=['date'])
    df = df.sort_values(['sid', 'date']).reset_index(drop=True)

    df = add_calendar_features(df)

    # 用训练时的类别集合做 LabelEncoder，保证训练/预测一致
    le = LabelEncoder()
    le.classes_ = np.array(checkpoint['category_classes'])
    df['category_code'] = le.transform(df['category'])

    for idx, row in df.iterrows():
        weather = get_weather(lat=row['latitude'], lon=row['longitude'], date_str=row['date'].strftime('%Y-%m-%d'))
        df.at[idx, 'temperature'] = weather['temperature']
        df.at[idx, 'weather_code'] = weather['weather_code']

    feature_names = checkpoint.get('feature_names')

    def build_sequences(df, feature_names, seq_len):
        X_seq = []
        dates_seq = []
        sids_seq = []
        for _sid, group in df.groupby('sid', sort=False):
            group = group.sort_values('date')
            X_group = group[feature_names].values
            dates_group = group['date'].values
            sid_group = group['sid'].values
            for i in range(seq_len, len(group)):
                X_seq.append(X_group[i-seq_len:i])
                dates_seq.append(dates_group[i])
                sids_seq.append(sid_group[i])
        return np.asarray(X_seq), dates_seq, sids_seq

    X_raw, dates, sids = build_sequences(df, feature_names, 7)
    if len(X_raw) == 0:
        raise ValueError("Not enough data to form sequences. Need at least seq_len rows per store.")

    # 标准化
    feature_scaler = checkpoint['feature_scaler']
    n_samples, seq_len_, n_feats = X_raw.shape
    X_flat = X_raw.reshape(-1, n_feats)
    X_flat_scaled = feature_scaler.transform(X_flat)
    X_scaled = X_flat_scaled.reshape(n_samples, seq_len_, n_feats)

    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    output_size = checkpoint['output_size']
    net = LSTMModel(input_size, hidden_size, num_layers, output_size)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred_scaled = net(X_tensor).numpy()

    y_pred = np.expm1(y_pred_scaled)

    target_names = checkpoint['target_names']
    pred_df = pd.DataFrame({
        'sid': sids,
        'date': dates
    })
    for i, name in enumerate(target_names):
        pred_df[name] = y_pred[:, i]

    print(pred_df.head(20) if len(pred_df) > 20 else pred_df)
    typer.echo(f"\nTotal predictions: {len(pred_df)}")

def main() -> None:
    app()
