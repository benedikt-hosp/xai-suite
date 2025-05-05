import logging
import os
import json
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.evaluate_with_selected_features import evaluate_with_selected_features
from src.utils.fix_placeholder_configs import fix_all_configs
from utils.project_paths import DATA_PROCESSED, resolve_paths, CONFIG_ROOT
import torch
from torch.utils.data import DataLoader

from src.utils.dataset_loader import get_dataset  # <-- du musst ggf. deinen Pfad anpassen
from src.utils.model_loader import get_model     # <-- du musst ggf. deinen Pfad anpassen
from src.xai.factory import get_xai_method   # <-- du musst ggf. deinen Pfad anpassen
from src.utils.save_modul import save_feature_ranking  # oben definierte Methode




EXPERIMENTS_ROOT = CONFIG_ROOT


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# EXPERIMENTS_ROOT = Path("experiments/configs")  # z.B./experiments/configs/robustvision/foval/intgrad_accuracy.yaml
# EXPERIMENTS_ROOT = Path(__file__).parent.parent / "experiments" / "configs"

logger.info(f"[INFO] Looking for experiments in: {EXPERIMENTS_ROOT}")
logger.info(f"[INFO] Found datasets: {os.listdir(EXPERIMENTS_ROOT)}")


def preprocess_and_save(dataset_config):
    logger.info("[PREPROCESS] Preprocessing and saving dataset...")

    dataset = get_dataset(dataset_config["name"], dataset_config["params"], skip_load=True)

    dataset.input_data = dataset.create_features(dataset.input_data)
    dataset.input_data = dataset.normalize_data(dataset.input_data)
    dataset.input_data = dataset.calculate_transformations_for_features(dataset.input_data)
    dataset.input_data = dataset.scale_target(dataset.input_data, isTrain=True)

    # Resolve save path outside of /src
    # save_path = Path(__file__).resolve().parent.parent / dataset_config["params"]["save_path"]
    # save_path = DATA_PROCESSED / dataset_config["name"]
    save_path = DATA_PROCESSED

    dataset.save_processed_data(save_path)

    logger.info(f"[PREPROCESS] Saved processed data to: {save_path}")

def run_all_experiments():
    logger.info("[START] Running experiment loop...")

    # for dataset_name in os.listdir(EXPERIMENTS_ROOT):
    for dataset_name in tqdm(os.listdir(EXPERIMENTS_ROOT), desc="Datasets"):

        dataset_path = EXPERIMENTS_ROOT / dataset_name
        if not dataset_path.is_dir():
            continue

        for model_name in os.listdir(dataset_path):
            model_path = dataset_path / model_name
            logger.info(f"[INFO] Scanning model path: {model_path}")
            if not model_path.is_dir():
                continue

            logger.info(f"[INFO] Files: {os.listdir(model_path)}")

            for method_file in os.listdir(model_path):
                if not method_file.endswith(".json"):
                    continue

                method_name = method_file.replace(".json", "")
                config_path = model_path / method_file

                logger.info(f"\n🔍 Running: {dataset_name} | {model_name} | {method_name}")

                with open(config_path) as f:
                    cfg = json.load(f)
                    cfg = resolve_paths(cfg, dataset_name, model_name, method_name)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Load components
                dataset = get_dataset(cfg["dataset"]["name"], cfg["dataset"]["params"])
                model = get_model(cfg["model"]["name"], cfg["model"]["params"]).to(device)
                xai = get_xai_method(cfg["xai"]["method"])
                feature_names = dataset.feature_names
                if not feature_names:
                    raise ValueError("Dataset has no feature names. Did preprocessing fail?")

                # dataloader = DataLoader(dataset, batch_size=cfg.get("batch_size", 16), shuffle=False)
                batch_size = cfg["model"]["params"].get("batch_size", 16)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                logger.info(f"[CONFIG] Loaded config from: {config_path}")
                logger.info(f"[CONFIG] Batch size: {batch_size}")

                logger.info("[XAI] Computing feature importances using XAI method...")
                importance_scores = xai(model, dataloader, feature_names, **cfg["xai"].get("params", {}))


                logger.info(f"[SAVE] Saving feature ranking for: {dataset_name} | {model_name} | {method_name}")
                output_dir = Path(cfg["eval"]["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)


                save_feature_ranking(dataset_name, model_name, method_name, importance_scores, output_dir)

                # save_feature_ranking(dataset_name, model_name, method_name, importance_scores)

CONFIG_ROOT = Path(__file__).parent.parent / "experiments" / "configs"
FEATURES_ABS = Path(__file__).parent.parent / "datasets" / "robustvision" / "features.json"



if __name__ == "__main__":
    # fix_all_configs()

    # Optional: Nur beim ersten Mal oder bei neuen Daten, damit sequenzen berechnet werden und nicht zur
    # laufzeit berechnet werden müssen
    with open(EXPERIMENTS_ROOT / "rv" / "foval" / "deepACTIF.json") as f:
        cfg = json.load(f)
    cfg = resolve_paths(cfg, dataset_name="rv", model_name="foval", method_name="intgrad_accuracy")

    preprocess_and_save(cfg["dataset"])

    try:
        run_all_experiments()
    except Exception as e:
        traceback.print_exc()
    #
    # # evaluate_all_rankings()
    # # Top-10 Features aus deinem Ranking laden
    # ranking_df = pd.read_csv("results/rv_foval_deepACTIF/deepACTIF_ranking.csv")
    # top_10 = ranking_df["feature"].tolist()[:10]
    #
    # # Evaluation mit nur den Top-10 (Keep)
    # mae_keep, rmse_keep = evaluate_with_selected_features(model, dataset, top_10, mode="keep")
    #
    # # Evaluation mit allen außer den Top-10 (Remove)
    # mae_remove, rmse_remove = evaluate_with_selected_features(model, dataset, top_10, mode="remove")
