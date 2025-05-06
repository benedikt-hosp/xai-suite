#!/usr/bin/env python3
import json
from pathlib import Path

from src.utils.project_paths import (
    CONFIG_ROOT,  # e.g. experiments/configs
    DATA_RAW,  # data/raw/rv_csvs
    DATA_PROCESSED,  # data/processed/robustvision
    FEATURE_CONFIG,  # datasets/robustvision/features.json
    FEATURE_EVAL_TASKS,  # where to write these tasks
    FEATURE_EVAL_PERFORMANCE_MEASURES, resolve_paths, FEATURE_RANKING_LISTS
)

TOPK_PERCENTAGES = [10, 20, 30, 40]
STRATEGIES       = ["keep", "remove"]

def create_eval_tasks():
    FEATURE_EVAL_TASKS.mkdir(parents=True, exist_ok=True)
    count = 0

    for datasets_dir in CONFIG_ROOT.iterdir():
        if not datasets_dir.is_dir(): continue
        ds = datasets_dir.name
        # print(ds)

        for model_dir in datasets_dir.iterdir():
            if not model_dir.is_dir(): continue
            mdl = model_dir.name
            # print(mdl)

            for exp_cfg in (model_dir).glob("*.json"):
                print(exp_cfg)
                method = exp_cfg.stem
                full_cfg = json.loads(exp_cfg.read_text())
                # print(full_cfg)

                model_block = full_cfg["model"]
                cfg = resolve_paths(full_cfg,
                    dataset_name=exp_cfg.parents[2].name,  # “rv”
                    model_name=exp_cfg.parents[1].name,  # “foval”
                    method_name=exp_cfg.stem)  # “deepACTIF”
                # print(cfg["eval"]["output_dir"])

                for topk in TOPK_PERCENTAGES:
                    for strat in STRATEGIES:
                        task = {
                            "dataset": {
                                "name": ds,
                                "params": {
                                    "root":            str(DATA_RAW),
                                    "save_path":       str(DATA_PROCESSED),
                                    "sequence_length": full_cfg["dataset"]["params"]["sequence_length"],
                                    "feature_config":  str(FEATURE_CONFIG),
                                    "load_processed":  True
                                }
                            },
                            "model": model_block,
                            "ranking_path": str(FEATURE_RANKING_LISTS / ds / mdl / f"{method}.json"
                            ),
                            "topk_percent": topk,
                            "strategy":    strat
                        }

                        fname = f"{ds}__{mdl}__{method}__{topk}__{strat}_feature_list.json"
                        (FEATURE_EVAL_TASKS / fname).write_text(json.dumps(task, indent=2))
                        count += 1

    print(f"✅ Created {count} evaluation tasks in {FEATURE_EVAL_TASKS}")


    #
    #
    #     ranking_dir = Path(cfg["eval"]["output_dir"])
    #     for rank_file in ranking_dir.glob(f"{cfg['dataset']['name']}__{cfg['model']['name']}__{cfg['xai']['method']}*.json"):
    #         for topk in (10,20,30,40):
    #             for strategy in ("keep","remove"):
    #                 task = {
    #                     "dataset": cfg["dataset"],
    #                     "model":   cfg["model"],
    #                     "ranking_path": str(rank_file),
    #                     "topk_percent": topk,
    #                     "strategy":     strategy
    #                 }
    #                 out = FEATURE_EVAL_TASKS / f"{exp_cfg.stem}__{topk}__{strategy}.json"
    #                 out.write_text(json.dumps(task, indent=2))
    #                 count += 1
    #
    # print(f"✅ Created {count} evaluation tasks in {FEATURE_EVAL_TASKS}")
