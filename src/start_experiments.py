import logging
import os
import json
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datasets.factory import get_dataset
from src.utils.evaluate_loocv import run_loocv
from src.utils.torchtrainer import TorchTrainer
# from src.utils.torchtrainer import build_trainer
from src.xai.factory import get_xai_method
from models.factory import get_model
from src.utils.save_modul import save_feature_ranking
from src.utils.generate_eval_tasks import create_eval_tasks
from src.utils.run_all_feature_evaluation_tasks import run_all_tasks
from utils.project_paths import DATA_PROCESSED, resolve_paths, CONFIG_ROOT

# ──────────────────────────────────────────────────────────────────────────────
EXPERIMENTS_ROOT = CONFIG_ROOT

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"[INFO] Looking for experiments in: {EXPERIMENTS_ROOT}")
logger.info(f"[INFO] Found datasets: {os.listdir(EXPERIMENTS_ROOT)}")


def preprocess_and_save(dataset_config):
    """
    One‐shot: read raw → filter to meta+target+input cols → full pipeline → dump to disk.
    """
    logger.info("[PREPROCESS] Starting full preprocessing…")
    ds = get_dataset(dataset_config["name"], dataset_config["params"])
    ds._full_preprocess_and_save() # dataset_config["params"]["root"], dataset_config["params"]["save_path"], dataset_config["params"]["feature_config"], dataset_config["params"]["sequence_length"])

    logger.info(f"[PREPROCESS] Cached processed data under: {dataset_config['params']['save_path']}")

def compute_ranked_lists_for_all_methods():
    logger.info("[START] Running experiment loop…")
    for dataset_name in tqdm(os.listdir(EXPERIMENTS_ROOT), desc="Datasets"):
        ds_dir = EXPERIMENTS_ROOT / dataset_name
        if not ds_dir.is_dir(): continue

        for model_name in os.listdir(ds_dir):
            mdl_dir = ds_dir / model_name
            if not mdl_dir.is_dir(): continue

            for method_file in os.listdir(mdl_dir):
                if not method_file.endswith(".json"): continue

                method_name = method_file[:-5]
                cfg = resolve_paths(
                    json.load(open(mdl_dir / method_file)),
                    dataset_name, model_name, method_name
                )
                logger.info(f"🔍  {dataset_name} | {model_name} | {method_name}")

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # 1) load *processed* dataset
                ds_params = deepcopy(cfg["dataset"]["params"])
                ds_params["load_processed"] = True
                raw_ds = get_dataset(cfg["dataset"]["name"], ds_params)
                # deep clone so we don't mutate the original
                ds_for_xai = deepcopy(raw_ds)

                # (optionally) subset features here:
                # ds_for_xai.set_selected_features(...)

                _, _, F = ds_for_xai.features_raw.shape
                feature_names = ds_for_xai.feature_names

                # 2) trainer‐factory
                def trainer_factory(input_dim, f_names=None):
                    model = get_model(
                        cfg["model"]["name"],
                        input_size=input_dim,
                        **cfg["model"]["params"]
                    ).to(device)

                    return TorchTrainer(
                        model=model,
                        optimizer_cfg={
                            "type": "adamw",
                            "lr":    cfg["model"]["params"]["learning_rate"],
                            "weight_decay": cfg["model"]["params"]["weight_decay"]
                        },
                        scheduler_cfg={
                            "type": "cosine",
                            "T_max": cfg["model"]["params"].get("epochs", 300),
                            "eta_min": 0.0
                        },
                        loss_cfg={"type": "smoothl1", "beta": 0.75},
                        early_stopping_cfg={"patience": 150, "min_delta": 0.0},
                        device=device
                    )

                # 3) run LOOCV *for metrics* (optional; you can skip if you only want XAI)
                subject_mae, mean_mae = run_loocv(trainer_factory, ds_for_xai)
                logger.info(f"[RESULT] LOOCV train MAE (per-subject): {subject_mae}")
                logger.info(f"[RESULT] LOOCV train mean MAE: {mean_mae:.4f}")

                # 4) now compute XAI attributions fold-by-fold
                xai = get_xai_method(cfg["xai"]["method"])
                fold_imps = []
                for held in ds_for_xai.subject_list:
                    # train on N−1
                    tr_ld, val_ld = ds_for_xai.get_fold(held)
                    # get a fresh trainer & train it
                    trainer = trainer_factory(F, feature_names)
                    trainer.set_dataloaders(tr_ld, val_ld)
                    trainer.fit(epochs=cfg["model"]["params"].get("epochs", 300))

                    # explain the held-out window
                    scores = xai(trainer.model, val_ld, feature_names,
                                 **cfg["xai"].get("params", {}))
                    fold_imps.append(scores)

                # 5) aggregate and save
                importance_scores = np.mean(fold_imps, axis=0)
                save_feature_ranking(dataset_name, model_name, method_name, importance_scores)


if __name__ == "__main__":
    # 1) Precompute & cache dataset once
    with open(EXPERIMENTS_ROOT / "rv" / "foval" / "deepACTIF.json") as f:
        cfg = resolve_paths(json.load(f), dataset_name="rv", model_name="foval", method_name="intgrad_accuracy")
    preprocess_and_save(cfg["dataset"])

    # 2) Run all your XAI experiments
    try:
        compute_ranked_lists_for_all_methods()
    except Exception:
        traceback.print_exc()

    # # 3) Kick off any downstream tasks you’ve automated
    # create_eval_tasks()
    # run_all_tasks()
