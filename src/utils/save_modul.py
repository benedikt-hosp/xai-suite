import os
import json
from pathlib import Path


# def save_feature_ranking(dataset_name, model_name, method_name, feature_names, scores):
#     os.makedirs(f"experiments/rankings/{dataset_name}/{model_name}", exist_ok=True)
#     out_path = f"experiments/rankings/{dataset_name}/{model_name}/{method_name}.json"
#     ranking = [{"feature": f, "score": float(s)} for f, s in sorted(zip(feature_names, scores), key=lambda x: -x[1])]
#     print(f"[SAVE] Writing ranking to: {out_path}")
#
#     with open(out_path, "w") as f:
#         json.dump(ranking, f, indent=2)
#     print(f"✅ Saved ranking to {out_path}")

# utils/save_modul.py
# def save_feature_ranking(dataset, model, method, importance_scores):
#     save_dir = Path("results") / dataset / model
#     save_dir.mkdir(parents=True, exist_ok=True)
#
#     # `importance_scores` ist bereits eine Liste von dicts
#     save_path = save_dir / f"{method}_ranking.json"
#     with open(save_path, "w") as f:
#         json.dump(importance_scores, f, indent=2)

def save_feature_ranking(dataset_name, model_name, method_name, scores, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{method_name}_ranking.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=4)
