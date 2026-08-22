import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def prepare_data(
    df: pd.DataFrame, target_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    df = df.copy()
    sort_cols = ["date"] + (["sid"] if "sid" in df.columns else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    le_weather = LabelEncoder()
    le_category = LabelEncoder()
    df["weather_code"] = le_weather.fit_transform(df["weather"])
    df["category_code"] = le_category.fit_transform(df["category"])

    features = [
        "area",
        "longitude",
        "latitude",
        "temperature",
        "oil_price",
        "weekday",
        "month",
        "day_of_year",
        "weather_code",
        "category_code",
    ]
    X = df[features]
    y = df[target_cols]

    # 按日期划分训练和测试集
    split_date = df["date"].quantile(0.8)
    train_mask = df["date"] <= split_date
    test_mask = ~train_mask
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    # 数值特征做标准化
    num_cols = [
        "area",
        "longitude",
        "latitude",
        "temperature",
        "oil_price",
        "weekday",
        "month",
        "day_of_year",
    ]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test, features


def prepare_lstm_data(
    df: pd.DataFrame, target_cols: list[str], seq_len: int = 7, train_ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    按店铺和时间构建序列，并按日期划分训练集和测试集。

    测试集的第一个窗口可以使用切分日期之前的 ``seq_len`` 天作为历史上下文，
    但测试目标本身不会参与训练或输入窗口。
    """
    df = df.copy()
    df = df.sort_values(["sid", "date"]).reset_index(drop=True)
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    le_weather = LabelEncoder()
    le_category = LabelEncoder()
    df["weather_code"] = le_weather.fit_transform(df["weather"])
    df["category_code"] = le_category.fit_transform(df["category"])

    features = [
        "area",
        "longitude",
        "latitude",
        "temperature",
        "oil_price",
        "weekday",
        "month",
        "day_of_year",
        "weather_code",
        "category_code",
    ]

    split_date = df["date"].quantile(train_ratio)
    train_mask = df["date"] <= split_date
    test_mask = ~train_mask
    train_df = df.loc[train_mask]

    scaler = StandardScaler()
    scaler.fit(train_df[features].values)

    def create_sequences(target_mask: pd.Series):
        X_seq, y_seq = [], []
        for _sid, group in df.groupby("sid", sort=False):
            group = group.sort_values("date")
            X_group = group[features].values
            y_group = group[target_cols].values
            dates_group = group["date"].values
            for i in range(seq_len, len(group)):
                # 窗口只到目标日前一天；测试目标使用训练期历史上下文。
                if target_mask.iloc[group.index[i]]:
                    X_seq.append(X_group[i - seq_len : i])
                    y_seq.append(y_group[i])
        return np.asarray(X_seq), np.asarray(y_seq)

    X_train_raw, y_train = create_sequences(train_mask)
    X_test_raw, y_test = create_sequences(test_mask)

    def scale_3d(X_raw):
        n_samples, seq_len, n_features = X_raw.shape
        X_flat = X_raw.reshape(-1, n_features)
        X_flat_scaled = scaler.transform(X_flat)
        return X_flat_scaled.reshape(n_samples, seq_len, n_features)

    X_train = scale_3d(X_train_raw)
    X_test = scale_3d(X_test_raw)

    return X_train, y_train, X_test, y_test


# 实际数据中，passby和其他数据不在一个量级，target也要做标准化
def scale_targets(y_train, y_test):
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_test_scaled = scaler.transform(y_test)
    return y_train_scaled, y_test_scaled, scaler
