from typing import Annotated
import typer
from flow_predictor.fake import fake_all

app = typer.Typer()

@app.command()
def cli(fake: Annotated[bool, typer.Option(help="Generate fake data.")] = False, train: Annotated[bool, typer.Option(help="Start training.")] = False):
    if fake:
        df = fake_all()
        print(f"Fake data: {df.describe()}")
    if train:
        print("train")

def main() -> None:
    app()
