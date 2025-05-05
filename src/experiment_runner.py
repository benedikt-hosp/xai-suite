import os
import yaml
import torch
import json
import numpy as np
from torch.utils.data import DataLoader

from src.utils.config_loader import load_feature_config
from src.utils.feature_loader import load_feature_config  # falls du das extern gemacht hast

def get_dataset(name, params):
    feature_config_path = params.get("feature_config")
    if not feature_config_path:
        raise ValueError("Missing 'feature_config' in dataset params.")

    input_features, target_feature, meta_features = load_feature_config(feature_config_path)

    if name == "rv":
        from datasets.robustvision import RobustVisionDataset
        return RobustVisionDataset(
            root=params["root"],
            split=params["split"],
            input_features=input_features,
            target_feature=target_feature,
            meta_features=meta_features
        )

    raise ValueError(f"Unknown dataset name '{name}'")



def get_model(name, params):
    if name == "foval":
        from FOVAL import FOVAL
        return FOVAL(**params)
    else:
        raise ValueError(f"Unknown model: {name}")

def get_xai_method(name):
    if name == "integrated_gradients":
        from xai.intgrad import compute_integrated_gradients
        return compute_integrated_gradients
    elif name == "deeplift":
        pass  # TODO
    elif name == "shap":
        pass  # TODO
    else:
        raise ValueError(f"Unknown XAI method: {name}")

# ---- FEATURE RANKING STORAGE ----
def save_feature_ranking(ranking, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ranking, f, indent=2)

# ---- MAIN LOOP ----
EXPERIMENT_ROOT = "configs/experiments"
RANKING_OUTPUT_DIR = "experiments/rankings"

for config_file in os.listdir(EXPERIMENT_ROOT):
    if not config_file.endswith(".yaml"):
        continue

    cfg_path = os.path.join(EXPERIMENT_ROOT, config_file)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = get_dataset(cfg["dataset"]["name"], cfg["dataset"]["params"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = get_model(cfg["model"]["name"], cfg["model"]["params"])
    model.eval()
    model.to(cfg["eval"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    xai_method = get_xai_method(cfg["xai"]["method"])

    # Feature attribution pooling
    attr_pool = []
    for x, y in dataloader:
        x = x.to(model.device)
        y_label = y.item()
        attr = xai_method(model, x, y_label, **cfg["xai"]["params"])
        attr_pool.append(attr.detach().cpu().squeeze(0).numpy())

    # Aggregation: mean absolute attribution per feature
    attr_array = np.stack(attr_pool)  # [samples, timesteps, features]
    feature_importance = np.mean(np.abs(attr_array), axis=(0, 1))  # [features]

    # Rank and store
    feature_ranking = {
        "ranking": [int(i) for i in np.argsort(-feature_importance)],
        "importance": feature_importance.tolist()
    }

    ranking_outfile = os.path.join(
        RANKING_OUTPUT_DIR,
        cfg["dataset"]["name"],
        cfg["model"]["name"],
        cfg["xai"]["method"] + ".json"
    )
    save_feature_ranking(feature_ranking, ranking_outfile)
    print(f"Saved ranking to: {ranking_outfile}")
