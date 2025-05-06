import os
import json
from glob import glob
from src.utils.project_paths import EXPERIMENT_RESULTS


def generate_feature_selection_tasks():
    """
    Generate feature selection tasks based on existing feature rankings.
    """

    # Einstellungen
    topk_percents = [10, 20, 30, 40]
    strategies = ["keep", "remove"]
    datasets = ["rv", "giw", "tufts"]
    ranking_root = EXPERIMENT_RESULTS
    output_config_dir = "results/feature_selections_performance"


    # Überprüfen, ob die Verzeichnisse existieren
    if not os.path.exists(ranking_root):
        print("[ERROR] 'feature_rankings' directory does not exist.")
        return

    if not os.path.exists(output_config_dir):
        print("[ERROR] 'experiments/results' directory does not exist.")
        os.makedirs(output_config_dir, exist_ok=True)


    task_id = 0
    n_tasks = 0

    for dataset in datasets:
        dataset_path = os.path.join(ranking_root, dataset)
        if not os.path.exists(dataset_path):
            print(f"[WARN] Dataset folder not found: {dataset_path}")
            continue

        for model in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model)
            if not os.path.isdir(model_path):
                continue

            for method_file in glob(f"{model_path}/*.json"):
                method = os.path.splitext(os.path.basename(method_file))[0]

                for k in topk_percents:
                    for strat in strategies:
                        task = {
                            "dataset": dataset,
                            "model": model,
                            "xai_method": method,
                            "topk_percent": k,
                            "strategy": strat,
                            "ranking_path": method_file,
                            "result_path": f"experiments/results/{dataset}/{model}/{method}_{strat}_top{k}.json"
                        }

                        task_id += 1
                        task_file = os.path.join(output_config_dir, f"task_{task_id:04d}.json")
                        with open(task_file, "w") as f:
                            json.dump(task, f, indent=2)
                        n_tasks += 1

    print(f"[INFO] Saved {n_tasks} tasks to {output_config_dir}")
