from copy import deepcopy
import json
import logging

import torch
from models.factory        import get_model
from src.utils.torchtrainer import TorchTrainer
from src.utils.evaluate_loocv import run_loocv
from datasets.factory      import get_dataset

logger = logging.getLogger(__name__)

def evaluate_with_selected_features(cfg):
    # --- 1) load ranking list ---
    ranking_path = cfg["ranking_path"]
    with open(ranking_path, "r") as f:
        rankings = json.load(f)

    total = len(rankings)
    top_k = int(cfg["topk_percent"] / 100 * total)
    selected = [r["feature"] for r in rankings[:top_k]]

    # --- 2) load dataset from cache ---
    ds_params = deepcopy(cfg["dataset"]["params"])
    ds_params["load_processed"] = True
    dataset = get_dataset(cfg["dataset"]["name"], ds_params)

    # --- 3) subset/drop features ---
    dataset.set_selected_features(selected, mode=cfg["strategy"])
    logger.info(f"Kept {len(dataset.feature_names)} features")

    # --- 4) build model and trainer_factory ---
    # instantiate a fresh model for each fold
    def trainer_factory(input_dim, feature_names=None):
        # 1) build the module
        model = get_model(
            cfg["model"]["name"],
            input_size=input_dim,
            **cfg["model"]["params"]
        )
        # 2) wrap in TorchTrainer
        trainer = TorchTrainer(
            model=model,
            optimizer_cfg={
                "type": "adamw",
                "lr": cfg["model"]["params"]["learning_rate"],
                "weight_decay": cfg["model"]["params"]["weight_decay"]
            },
            scheduler_cfg={
                "type": "cosine",
                "T_max": cfg["model"]["params"].get("epochs", 300),
                "eta_min": 0.0
            },
            loss_cfg={ "type": "smoothl1", "beta": 0.75 },
            early_stopping_cfg={ "patience": 150, "min_delta": 0.0 },
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return trainer

    logger.info("Starting LOOCV …")
    subject_mae, mean_mae = run_loocv(trainer_factory, dataset)
    logger.info(f"[RESULT] LOOCV complete — mean_mae={mean_mae:.4f}")

    return {
        "subject_mae": subject_mae,
        "mean_mae": mean_mae
    }
