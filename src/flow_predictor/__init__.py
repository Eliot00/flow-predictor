from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Annotated
import pandas as pd
import typer
from flow_predictor.fake import fake_all

app = typer.Typer()

@app.command()
def cli(fake: Annotated[bool, typer.Option(help="Generate fake data.")] = False, train: Annotated[bool, typer.Option(help="Start training.")] = False):
    if fake:
        df = fake_all()
        print(f"Fake data: {df.describe()}")
    if train:
        df = fake_all()

        # 特征工程
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
        y = df['passby_visit']

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

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)

        print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
        print(f"R2: {r2_score(y_test, y_pred):.2f}")

        # 系数
        coef_df = pd.DataFrame({
            'feature': features,
            'coefficient': model.coef_
        }).sort_values('coefficient', ascending=False)
        print(coef_df)

def main() -> None:
    app()
