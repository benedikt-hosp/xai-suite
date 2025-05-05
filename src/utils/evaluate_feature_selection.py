import os
import pandas as pd
import json
import torch
from tqdm import tqdm
from pathlib import Path
from src.utils.dataset_loader import get_dataset
from src.utils.model_loader import get_model
from src.utils.evaluate_with_selected_features import evaluate_with_selected_features  # Du musst diese Funktion ggf. erstellen
from src.utils.project_paths import CONFIG_ROOT, PROJECT_ROOT

# 🔧 Absoluter Pfad zum "results"-Verzeichnis
RESULTS_DIR = PROJECT_ROOT / "results"

# K%-Stufen für Evaluation
K_VALUES = [10, 20, 30, 40]  # Prozent der Features

def evaluate_all_rankings():
    for dataset_name in os.listdir(RESULTS_DIR):
        dataset_path = RESULTS_DIR / dataset_name
        for model_name in os.listdir(dataset_path):
            model_path = dataset_path / model_name
            for method_name in os.listdir(model_path):
                result_path = model_path / method_name / "feature_ranking.csv"

                if not result_path.exists():
                    print(f"⛔ Kein Ranking gefunden: {result_path}")
                    continue

                print(f"📊 Evaluating: {dataset_name} | {model_name} | {method_name}")
                # Lade Feature-Ranking
                ranking_df = pd.read_csv(result_path)
                feature_ranks = ranking_df["feature"].tolist()

                # Lade zugehörige Konfiguration
                config_file = CONFIG_ROOT / dataset_name / model_name / f"{method_name}.json"
                if not config_file.exists():
                    print(f"⚠️ Keine Konfig gefunden: {config_file}")
                    continue
                with open(config_file) as f:
                    cfg = json.load(f)

                all_results = []

                for k in K_VALUES:
                    keep_top_n = max(1, int(len(feature_ranks) * k / 100))
                    selected_features = feature_ranks[:keep_top_n]

                    print(f" → {k}% ({keep_top_n} Features): {selected_features[:3]}...")

                    # Setze reduzierte Feature-Liste
                    cfg["dataset"]["params"]["selected_features"] = selected_features
                    dataset = get_dataset(cfg["dataset"]["name"], cfg["dataset"]["params"])
                    model = get_model(cfg["model"]["name"], cfg["model"]["params"])

                    mae, rmse = evaluate_with_selected_features(model, dataset)
                    all_results.append({
                        "method": method_name,
                        "k_percent": k,
                        "kept_features": keep_top_n,
                        "mae": mae,
                        "rmse": rmse
                    })

                # Speichere die Ergebnisse
                output_df = pd.DataFrame(all_results)
                save_path = model_path / method_name / "evaluation_results.csv"
                output_df.to_csv(save_path, index=False)
                print(f"✅ Gespeichert: {save_path}")
