# src/utils/evaluate_with_selected_features.py

import json
import logging
from copy import deepcopy

from datasets.factory        import get_dataset
from src.utils.evaluate_loocv import run_loocv
from src.utils.torchtrainer   import build_trainer

# configure a module‐level logger
logger = logging.getLogger(__name__)


def evaluate_with_selected_features(cfg):
    """
    1) Load a saved feature‐ranking JSON
    2) Re-load your preprocessed tensor cache
    3) Subset or drop features based on strategy + topk%
    4) Build a fresh Trainer factory and run LOOCV
    """
    logger.debug(f"→ evaluate_with_selected_features called with:\n{json.dumps(cfg, indent=2)}")

    # --- 1) load ranking list ---
    ranking_path = cfg["ranking_path"]
    logger.info(f"Loading feature ranking from: {ranking_path}")
    with open(ranking_path, "r") as f:
        rankings = json.load(f)

    total = len(rankings)
    logger.info(f" → Found {total} features in ranking")
    if total == 0:
        logger.error("No features in ranking – aborting")
        raise ValueError(f"No features in ranking: {ranking_path}")

    top_k = int(cfg["topk_percent"] / 100 * total)
    logger.info(f" → topk_percent={cfg['topk_percent']}%, selecting top {top_k} features")
    if top_k < 1:
        logger.error("topk_percent too small – aborting")
        raise ValueError(f"topk_percent too small: {cfg['topk_percent']}% of {total}")

    # --- 2) load dataset from cache ---
    ds_name   = cfg["dataset"]["name"]
    ds_params = deepcopy(cfg["dataset"]["params"])
    ds_params["load_processed"] = True
    logger.info(f"Instantiating dataset '{ds_name}' with params: {ds_params}")
    dataset = get_dataset(ds_name, ds_params)

    # --- 3) subset/drop features ---
    selected = [r["feature"] for r in rankings[:top_k]]
    logger.debug(f"Selected features list ({len(selected)}): {selected}")
    strategy = cfg["strategy"]
    dataset.set_selected_features(selected, mode=strategy)
    logger.info(f"After '{strategy}' mode dataset now has {len(dataset.feature_names)} features")

    # --- 4) build trainer & run LOOCV ---
    logger.info(f"Building trainer factory with model config: {cfg['model']}")
    trainer_factory = build_trainer(cfg["model"])
    logger.info("Starting LOOCV …")
    subject_mae, mean_mae = run_loocv(trainer_factory, dataset)
    logger.info(f"[RESULT] LOOCV complete — mean_mae={mean_mae:.4f}")

    # return raw metrics
    return {
        "subject_mae": subject_mae,
        "mean_mae":   mean_mae
    }
