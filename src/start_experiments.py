import logging
import os
import json
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.factory import get_dataset
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
    ds._full_preprocess_and_save()

    logger.info(f"[PREPROCESS] Cached processed data under: {dataset_config['params']['save_path']}")


def compute_ranked_lists_for_all_methods():
    logger.info("[START] Running experiment loop…")
    for dataset_name in tqdm(os.listdir(EXPERIMENTS_ROOT), desc="Datasets"):
        dataset_path = EXPERIMENTS_ROOT / dataset_name
        if not dataset_path.is_dir():
            continue

        for model_name in os.listdir(dataset_path):
            model_path = dataset_path / model_name
            if not model_path.is_dir():
                continue

            for method_file in os.listdir(model_path):
                if not method_file.endswith(".json"):
                    continue

                method_name = method_file[:-5]
                config_path = model_path / method_file
                logger.info(f"🔍  {dataset_name} | {model_name} | {method_name}")

                # load & resolve paths
                with open(config_path) as f:
                    cfg = resolve_paths(json.load(f), dataset_name, model_name, method_name)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                cfg["load_processed"] = True

                # load the already‐processed dataset (it will load .pt files)
                dataset = get_dataset(cfg["dataset"]["name"], cfg["dataset"]["params"])
                # we just need the number of features, not loaders
                input_size = dataset.features_tensor.shape[2]

                # init model & XAI
                model = get_model(cfg["model"]["name"], input_size=input_size, **cfg["model"]["params"]).to(device)
                xai = get_xai_method(cfg["xai"]["method"])
                feature_names = dataset.feature_names

                # compute importances
                batch_size = cfg["model"]["params"].get("batch_size", 16)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                importance_scores = xai(model, loader, feature_names, **cfg["xai"].get("params", {}))

                # save ranking
                save_feature_ranking(dataset_name, model_name, method_name, importance_scores)


if __name__ == "__main__":
    # 1) Precompute & cache dataset once
    # with open(EXPERIMENTS_ROOT / "rv" / "foval" / "deepACTIF.json") as f:
    #     cfg = resolve_paths(json.load(f), dataset_name="rv", model_name="foval", method_name="intgrad_accuracy")
    # preprocess_and_save(cfg["dataset"])
    #
    # # 2) Run all your XAI experiments
    # try:
    #     compute_ranked_lists_for_all_methods()
    # except Exception:
    #     traceback.print_exc()

    # 3) Kick off any downstream tasks you’ve automated
    create_eval_tasks()
    run_all_tasks()
