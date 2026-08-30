import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import lightning as L
import numpy as np
import pandas as pd
import torch
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import (
    EarlyStopping,
    EarlyStoppingReason,
)
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader, TensorDataset

from flow_predictor.fake import fake_all
from flow_predictor.network import LSTMModel, LtModule
from flow_predictor.prepare import (
    add_calendar_features,
    encode_category,
    prepare_data,
    prepare_lstm_data,
    split_by_date,
    transform_targets,
)
from flow_predictor.utils import get_weather, set_seed

app = typer.Typer()

TARGET_COLS = [
    "passby_visit",
    "entering_people",
    "dwell_people",
    "served_people",
]


# 业务时区：国内店铺用东八区；国外店铺的数据采集层应统一转换到此时区再入库
BUSINESS_TZ = ZoneInfo("Asia/Shanghai")


def report_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    for i, col in enumerate(TARGET_COLS):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{col}: MSE={mse:.2f}, R²={r2:.2f}")


@app.command()
def prepare():
    # 临时模拟
    start_date = datetime(2026, 1, 1, tzinfo=BUSINESS_TZ)
    end_date = datetime.now(tz=BUSINESS_TZ) - timedelta(days=1)

    current = start_date
    while current <= end_date:
        get_weather(lon=121.4455, lat=31.2264, date_str=current.strftime("%Y-%m-%d"))
        time.sleep(1)
        current += timedelta(days=1)


@app.command()
def train(
    data_file: Annotated[Path | None, typer.Option(help="Real data file path")] = None,
    model: Annotated[
        str, typer.Option(help="Model type: 'lstm', 'rf', 'gbr' or 'naive'")
    ] = "lstm",
    max_epochs: Annotated[
        int,
        typer.Option(
            help="Maximum training epochs (early stopping usually ends training first)"
        ),
    ] = 1000,
    patience: Annotated[
        int,
        typer.Option(
            help="Early stopping patience: val epochs without val_loss improvement"
        ),
    ] = 10,
    batch_size: Annotated[int, typer.Option(help="Batch size for training")] = 32,
    hidden_size: Annotated[int, typer.Option(help="Number of LSTM hidden units")] = 64,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 0.001,
    lag: Annotated[
        str | None, typer.Option(help="Comma-separated lag days, e.g. '1,7'")
    ] = None,
    save_path: Annotated[
        Path | None,
        typer.Option(help="Directory or file path to save the trained model"),
    ] = None,
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

    if model == "lstm":
        X_train, y_train, X_test, y_test, features, scaler = prepare_lstm_data(
            df, TARGET_COLS
        )
        y_train_log, y_test_log = transform_targets(y_train, y_test)
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_log, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_log, dtype=torch.float32)

        net = LSTMModel(
            X_train.shape[2],
            hidden_size=hidden_size,
            num_layers=2,
            output_size=y_train_log.shape[1],
        )
        module = LtModule(
            net=net,
            lr=lr,
            meta={
                "model_type": "lstm",
                "feature_names": features,
                "target_names": TARGET_COLS,
                "feature_mean": scaler.mean_.tolist(),
                "feature_scale": scaler.scale_.tolist(),
                "net_args": {
                    "input_size": X_train.shape[2],
                    "hidden_size": hidden_size,
                    "num_layers": 2,
                    "output_size": y_train_log.shape[1],
                },
            },
        )

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(X_test_t, y_test_t), batch_size=batch_size
        )

        ckpt = None
        if save_path:
            save_path = Path(save_path)
            if save_path.is_dir() or save_path.suffix == "":
                save_path = (
                    save_path
                    / f"lstm_model_{datetime.now(tz=BUSINESS_TZ).strftime('%Y%m%d_%H%M%S')}.ckpt"
                )
            ckpt = ModelCheckpoint(
                dirpath=save_path.parent,
                filename=save_path.stem,
                monitor="val_loss",
                mode="min",
            )

        es = EarlyStopping(
            monitor="val_loss", mode="min", patience=patience, check_finite=True
        )
        callbacks = [es] + ([ckpt] if ckpt else [])

        trainer = L.Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=1.0,
            callbacks=callbacks,
        )
        trainer.fit(module, train_loader, val_loader)

        if es.stopping_reason == EarlyStoppingReason.PATIENCE_EXHAUSTED:
            typer.echo(
                f"Early stopped at epoch {es.stopped_epoch + 1} "
                f"(no val_loss improvement for {patience} checks)"
            )
        elif es.stopping_reason == EarlyStoppingReason.NON_FINITE_METRIC:
            typer.echo("Early stopped: val_loss became NaN/inf")

        best_module = (
            LtModule.load_from_checkpoint(
                ckpt.best_model_path,
                map_location="cpu",
                net=LSTMModel(
                    X_train.shape[2],
                    hidden_size=hidden_size,
                    num_layers=2,
                    output_size=y_train_log.shape[1],
                ),
            )
            if ckpt
            else module
        )
        best_module.eval()
        with torch.no_grad():
            y_pred_scaled = best_module.net(X_test_t).numpy()
        y_pred = np.expm1(y_pred_scaled)
        y_true = np.expm1(y_test_log)
        report_metrics(y_true, y_pred)

        if ckpt:
            typer.echo(
                f"Best checkpoint (val_loss={float(ckpt.best_model_score):.4f}) "
                f"saved to {ckpt.best_model_path}"
            )
    elif model in {"rf", "gbr"}:
        lag_days = [int(x.strip()) for x in lag.split(",")] if lag else None
        X_train_scaled, X_test_scaled, y_train, y_test, _, _ = prepare_data(
            df, TARGET_COLS, lag_days
        )
        y_train_log, y_test_log = transform_targets(y_train.values, y_test.values)

        if model == "rf":
            estimator = RandomForestRegressor(
                n_estimators=300, n_jobs=-1, random_state=42
            )
        else:
            estimator = MultiOutputRegressor(
                HistGradientBoostingRegressor(max_iter=300, random_state=42),
                n_jobs=-1,
            )

        estimator.fit(X_train_scaled, y_train_log)

        y_pred = np.expm1(estimator.predict(X_test_scaled))
        report_metrics(np.expm1(y_test_log), y_pred)

    elif model == "naive":
        df = df.sort_values(["sid", "date"]).reset_index(drop=True)
        _train_mask, test_mask = split_by_date(df)
        test = df.loc[test_mask]
        for target in TARGET_COLS:
            snaive = df.groupby("sid")[target].shift(7)
            y_true = test[target].values
            y_pred = snaive.loc[test.index].values
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            print(f"{target}: MSE={mse:.2f}, R²={r2:.2f}")
    else:
        typer.echo(f"Unknown model: {model} (expected 'lstm', 'rf', 'gbr' or 'naive')")


@app.command()
def predict(
    model_file: Annotated[
        Path, typer.Option(help="Saved Lightning checkpoint (.ckpt)")
    ],
    data_file: Annotated[Path, typer.Option(help="Stores, dates, features CSV file")],
):
    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    meta = checkpoint["hyper_parameters"]["meta"]
    df = _load_prediction_frame(data_file)

    feature_names = meta["feature_names"]

    def build_sequences(df, feature_names, seq_len):
        X_seq = []
        dates_seq = []
        sids_seq = []
        for _sid, group in df.groupby("sid", sort=False):
            group = group.sort_values("date")
            X_group = group[feature_names].values
            dates_group = group["date"].values
            sid_group = group["sid"].values
            for i in range(seq_len, len(group)):
                X_seq.append(X_group[i - seq_len : i])
                dates_seq.append(dates_group[i])
                sids_seq.append(sid_group[i])
        return np.asarray(X_seq), dates_seq, sids_seq

    X_raw, dates, sids = build_sequences(df, feature_names, 7)
    if len(X_raw) == 0:
        raise ValueError(
            "Not enough data to form sequences. Need at least seq_len rows per store."
        )

    # 标准化：用训练时保存的 mean/scale 手动还原 StandardScaler.transform
    feature_mean = np.array(meta["feature_mean"])
    feature_scale = np.array(meta["feature_scale"])
    feature_scale[feature_scale == 0] = 1.0
    n_samples, seq_len_, n_feats = X_raw.shape
    X_flat = X_raw.reshape(-1, n_feats)
    X_flat_scaled = (X_flat - feature_mean) / feature_scale
    X_scaled = X_flat_scaled.reshape(n_samples, seq_len_, n_feats)

    net = LSTMModel(**meta["net_args"])
    module = LtModule.load_from_checkpoint(model_file, map_location="cpu", net=net)
    module.eval()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred_scaled = module.net(X_tensor).numpy()

    y_pred = np.expm1(y_pred_scaled)

    target_names = meta["target_names"]
    pred_df = pd.DataFrame({"sid": sids, "date": dates})
    for i, name in enumerate(target_names):
        pred_df[name] = y_pred[:, i]

    print(pred_df.head(20) if len(pred_df) > 20 else pred_df)
    typer.echo(f"\nTotal predictions: {len(pred_df)}")


def _load_prediction_frame(data_file: Path) -> pd.DataFrame:
    df = pd.read_csv(data_file, parse_dates=["date"])
    df = df.sort_values(["sid", "date"]).reset_index(drop=True)

    df = add_calendar_features(df)

    df["category_code"] = encode_category(df)

    for idx, row in df.iterrows():
        weather = get_weather(
            lat=row["latitude"],
            lon=row["longitude"],
            date_str=row["date"].strftime("%Y-%m-%d"),
        )
        df.at[idx, "temperature"] = weather["temperature"]
        df.at[idx, "precipitation"] = weather["precipitation"]
        df.at[idx, "wind_speed"] = weather["wind_speed"]
        df.at[idx, "humidity"] = weather["humidity"]
        df.at[idx, "weather_code"] = weather["weather_code"]
    return df


def main() -> None:
    app()
