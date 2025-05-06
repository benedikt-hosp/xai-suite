import os
import json
import traceback
from pathlib import Path
from src.utils.evaluate_with_selected_features import evaluate_with_selected_features
from src.utils.project_paths import FEATURE_RANKING_LISTS, FEATURE_EVAL_TASKS, FEATURE_EVAL_PERFORMANCE_MEASURES

CONFIG_DIR = FEATURE_EVAL_TASKS
RESULTS_DIR = FEATURE_EVAL_PERFORMANCE_MEASURES
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_all_tasks():
    config_files = list(CONFIG_DIR.rglob("*.json"))

    print(CONFIG_DIR)
    print(RESULTS_DIR)
    print(f"[INFO] Found {len(config_files)} tasks to run.")

    for config_path in config_files:
        with open(config_path, "r") as f:
            cfg = json.load(f)

        result_file = RESULTS_DIR / config_path.with_suffix(".results.json").name
        if result_file.exists():
            print(f"[SKIP] Already done: {result_file.name}")
            continue

        print(f"[RUN] {config_path.name}")
        try:
            metrics = evaluate_with_selected_features(cfg)

            # if cfg.get("evaluate_feature_selection", False):
            #     from src.utils.evaluate_feature_selection import run_feature_selection_evaluation
            #     run_feature_selection_evaluation(
            #         dataset=dataset,
            #         model_cfg=cfg["model"],
            #         importance_scores=importance_scores,
            #         feature_names=feature_names,
            #         output_dir=cfg["evaluation"]["output_dir"],
            #         top_k_percents=cfg["evaluation"]["top_k_percents"],
            #         strategy=cfg["evaluation"]["strategy"],
            #         seed=cfg.get("seed", 42),
            #         subject_ids=dataset.subject_list
            #     )


            with open(result_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"✅ Saved result to {result_file}")
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Failed {config_path.name}: {e}")

