# src/utils/evaluate_with_selected_features.py

import json
from copy import deepcopy

from datasets.factory        import get_dataset
# from src.trainer_factory     import build_trainer
from src.utils.evaluate_loocv import run_loocv
from src.utils.torchtrainer import build_trainer


def evaluate_with_selected_features(cfg):
    """
    1) Load a saved feature‐ranking JSON
    2) Re-load your preprocessed tensor cache
    3) Subset or drop features based on strategy + topk%
    4) Build a fresh Trainer factory and run LOOCV
    """
    # --- 1) load ranking list ---
    with open(cfg["ranking_path"], "r") as f:
        rankings = json.load(f)
    total = len(rankings)
    if total == 0:
        raise ValueError(f"No features in ranking: {cfg['ranking_path']}")

    top_k = int(cfg["topk_percent"] / 100 * total)
    if top_k < 1:
        raise ValueError(f"topk_percent too small: {cfg['topk_percent']}% of {total}")

    # --- 2) load dataset from cache ---
    # your task JSON uses `dataset_name` & `dataset_params`
    ds_name   = cfg["dataset"]["name"]
    ds_params = deepcopy(cfg["dataset"]["params"])
    ds_params["load_processed"] = True

    dataset = get_dataset(ds_name, ds_params)

    # --- 3) subset/drop features ---
    selected = [r["feature"] for r in rankings[:top_k]]
    dataset.set_selected_features(selected, mode=cfg["strategy"])
    print(f"[INFO] Strategy={cfg['strategy']}, now using {len(dataset.feature_names)} features")

    # --- 4) build trainer & run LOOCV ---
    # note: pass both the dataset (for shape) and model cfg
    # trainer_factory = build_trainer(dataset)
    # subject_mae, mean_mae = run_loocv(trainer_factory, dataset)
    # trainer_factory = build_trainer(dataset)
    # subject_mae, mean_mae = run_loocv(trainer_factory, dataset)
    # 4. Build trainer factory (this returns a function expecting input_dim)
    trainer_factory = build_trainer(cfg["model"])
    subject_mae, mean_mae = run_loocv(trainer_factory, dataset)

    return {
        "subject_mae": subject_mae,
        "mean_mae":   mean_mae
    }
