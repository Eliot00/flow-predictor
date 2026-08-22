from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

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
