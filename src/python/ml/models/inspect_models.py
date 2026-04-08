from pathlib import Path
import joblib


MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    MODEL_DIR / "solar_xgboost_model.pkl",
    MODEL_DIR / "wind_xgboost_model.pkl",
    MODEL_DIR / "chance_model.pkl",
]


def describe_model(path: Path) -> None:
    model = joblib.load(path)
    feature_names = getattr(model, "feature_names_in_", None)

    print(f"Model: {path.name}")
    print(f"Type: {type(model).__name__}")
    print(f"Feature names: {list(feature_names) if feature_names is not None else 'Unavailable'}")
    print()


if __name__ == "__main__":
    for model_path in MODEL_PATHS:
        describe_model(model_path)
