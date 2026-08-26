import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_FEATURES = [
    "area",
    "longitude",
    "latitude",
    "temperature",
    "precipitation",
    "wind_speed",
    "humidity",
    "weekday",
    "month",
    "day_of_year",
    "is_workday",
    "is_holiday",
    "weather_code",
    "category_code",
]

NUMERIC_FEATURES = BASE_FEATURES[:-2]  # 排除两个 *_code 编码列


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加日历特征：weekday/month/day_of_year + 法定节假日/调休。"""
    df = df.copy()
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # chinese_calendar 的 is_workday 已包含调休逻辑：调休上班的周末返回 True
    import chinese_calendar as cn_cal

    dates = df["date"].dt.date
    df["is_workday"] = [int(cn_cal.is_workday(d)) for d in dates]
    df["is_holiday"] = [int(not cn_cal.is_workday(d)) for d in dates]
    return df


def add_time_and_encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加日历特征，并对 category 做 LabelEncoder 编码。

    weather_code是WMO标准天气代码
    """
    df = add_calendar_features(df)

    # TODO: 这块可能有多tags
    le_category = LabelEncoder()
    df["category_code"] = le_category.fit_transform(df["category"])
    return df


def split_by_date(df: pd.DataFrame, train_ratio: float = 0.8):
    """按日期分位数划分训练/测试掩码。"""
    split_date = df["date"].quantile(train_ratio)
    train_mask = df["date"] <= split_date
    return train_mask, ~train_mask


def prepare_data(
    df: pd.DataFrame, target_cols: list[str], lag_days: list[int] | None = None
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], StandardScaler
]:
    df = add_time_and_encode_features(df)
    sort_cols = ["date"] + (["sid"] if "sid" in df.columns else [])
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # MLP是没有利用时序信息的
    # 试验下，拿之前的targets充当现在的特征，也就是滞后特征
    if lag_days:
        for target in target_cols:
            for lag in lag_days:
                if "sid" in df.columns:
                    df[f"{target}_lag_{lag}"] = df.groupby("sid")[target].shift(lag)
                else:
                    df[f"{target}_lag_{lag}"] = df[target].shift(lag)
        # 前几行因为lag产生的nan删掉
        df = df.dropna()

    base_features = BASE_FEATURES
    if lag_days:
        lag_features = [f"{t}_lag_{lag}" for t in target_cols for lag in lag_days]
        features = base_features + lag_features
    else:
        features = base_features

    X = df[features]
    y = df[target_cols]

    # 按日期划分训练和测试集
    train_mask, test_mask = split_by_date(df)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    # 数值特征做标准化
    num_cols = NUMERIC_FEATURES[:]
    if lag_days:
        num_cols += [f"{t}_lag_{lag}" for t in target_cols for lag in lag_days]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test, features, scaler


def prepare_lstm_data(
    df: pd.DataFrame, target_cols: list[str], seq_len: int = 7, train_ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], StandardScaler]:
    """
    按店铺和时间构建序列，并按日期划分训练集和测试集。

    测试集的第一个窗口可以使用切分日期之前的 ``seq_len`` 天作为历史上下文，
    但测试目标本身不会参与训练或输入窗口。
    """
    df = add_time_and_encode_features(df)
    df = df.sort_values(["sid", "date"]).reset_index(drop=True)

    features = BASE_FEATURES

    train_mask, test_mask = split_by_date(df, train_ratio)
    train_df = df.loc[train_mask]

    scaler = StandardScaler()
    scaler.fit(train_df[features].values)

    def create_sequences(target_mask: pd.Series):
        X_seq, y_seq = [], []
        for _sid, group in df.groupby("sid", sort=False):
            group = group.sort_values("date")
            X_group = group[features].values
            y_group = group[target_cols].values
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

    return X_train, y_train, X_test, y_test, features, scaler


# 实际数据中，passby和其他数据不在一个量级，target用log1p变换：
# 压缩长尾、统一相对误差尺度，且 expm1 反变换后预测值恒为正
def transform_targets(y_train, y_test):
    return np.log1p(y_train), np.log1p(y_test)
