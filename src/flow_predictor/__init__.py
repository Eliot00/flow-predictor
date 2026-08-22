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
):
    if fake:
        df = fake_all()
        print(f"Fake data: {df.describe()}")
    if train:
        df = fake_all()

        X_train_scaled, X_test_scaled, y_train, y_test, features = prepare_data(df, ["passby_visit", "entering_people", "dwell_people", "served_people"])

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        
        for i, col in enumerate(y_train.columns):
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            print(f"{col}: MSE={mse:.2f}, R²={r2:.2f}")
    
        coef_df = pd.DataFrame(
            model.coef_,
            index=y_train.columns,
            columns=features
        )
        print("\n系数矩阵（行=目标，列=特征）:")
        print(coef_df)

def main() -> None:
    app()
