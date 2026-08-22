from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

def prepare_data(df: pd.DataFrame, target_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    df = df.copy()
    df['weekday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear

    le_waather = LabelEncoder()
    le_category = LabelEncoder()
    df['weather_code'] = le_waather.fit_transform(df['weather'])
    df['category_code'] = le_category.fit_transform(df['category'])

    features = ['area', 'longitude', 'latitude', 'temperature', 'oil_price', 
                'weekday', 'month', 'day_of_year', 'weather_code', 'category_code']
    X = df[features]
    y = df[target_cols]

    # 按时间划分训练和验证集
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 数值特征做标准化
    num_cols = ['area', 'longitude', 'latitude', 'temperature', 'oil_price', 
                'weekday', 'month', 'day_of_year']

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test, features

def prepare_lstm_data(df: pd.DataFrame, target_cols: list[str], seq_len: int = 7, train_ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    LSTM需要按店铺分组，按时间排序，然后构建滑动窗口
    """
    df = df.copy()
    df['weekday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear

    le_waather = LabelEncoder()
    le_category = LabelEncoder()
    df['weather_code'] = le_waather.fit_transform(df['weather'])
    df['category_code'] = le_category.fit_transform(df['category'])

    features = ['area', 'longitude', 'latitude', 'temperature', 'oil_price', 
                'weekday', 'month', 'day_of_year', 'weather_code', 'category_code']

    split_date = df['date'].quantile(train_ratio)
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()

    scaler = StandardScaler()

    train_features = train_df[features].values
    scaler.fit(train_features)

    def create_sequences(df_part):
        X_seq, y_seq = [], []
        for _sid, group in df_part.groupby('sid'):
            group = group.sort_values('date')
            X_group = group[features].values
            y_group = group[target_cols].values
            for i in range(len(group) - seq_len):
                X_seq.append(X_group[i:i+seq_len])
                y_seq.append(y_group[i+seq_len])
        return np.array(X_seq), np.array(y_seq)

    X_train_raw, y_train = create_sequences(train_df)
    X_test_raw, y_test = create_sequences(test_df)

    def scale_3d(X_raw):
        n_samples, seq_len, n_features = X_raw.shape
        X_flat = X_raw.reshape(-1, n_features)
        X_flat_scaled = scaler.transform(X_flat)
        return X_flat_scaled.reshape(n_samples, seq_len, n_features)

    X_train = scale_3d(X_train_raw)
    X_test = scale_3d(X_test_raw)

    return X_train, y_train, X_test, y_test
    
