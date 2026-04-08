import modal
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
SOLAR_MODEL_PATH = MODEL_DIR / "solar_xgboost_model.pkl"
WIND_MODEL_PATH = MODEL_DIR / "wind_xgboost_model.pkl"
CHANCE_MODEL_PATH = MODEL_DIR / "chance_model.pkl"

volume = modal.Volume.from_name("energy-models", create_if_missing=True)

with volume.batch_upload() as batch:
    batch.put_file(SOLAR_MODEL_PATH, "/models/solar_xgboost_model.pkl")
    batch.put_file(WIND_MODEL_PATH, "/models/wind_xgboost_model.pkl")
    batch.put_file(CHANCE_MODEL_PATH, "/models/chance_model.pkl")

print("Uploaded model files.")
