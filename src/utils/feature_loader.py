import json
from pathlib import Path

# def load_feature_config(path):
#     path = Path(path)
#     if not path.is_absolute():
#         path = Path(__file__).parent.parent / path
#     with open(path, "r") as f:
#         config = json.load(f)
#     return config["input_features"], config["target_feature"], config["meta_features"]


def load_feature_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent.parent / path  # ← Relativ zur Projektwurzel
    with open(path, "r") as f:
        config = json.load(f)
    return config["input_features"], config["target_feature"], config["meta_features"]
