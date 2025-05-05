import json
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "experiments" / "configs"

# Platzhalter und Felder, in denen sie vorkommen sollen
PLACEHOLDER_MAP = {
    "root": "__DATA_RAW__",
    "feature_config": "__FEATURE_CONFIG__",
    "save_path": "__DATA_PROCESSED__",
    "output_dir": "__RESULTS_DIR__"
}

def update_paths(config):
    updated = False

    # dataset params
    if "dataset" in config and "params" in config["dataset"]:
        for key in ["root", "feature_config", "save_path"]:
            if key in config["dataset"]["params"]:
                config["dataset"]["params"][key] = PLACEHOLDER_MAP[key]
                updated = True

    # eval output
    if "eval" in config and "output_dir" in config["eval"]:
        config["eval"]["output_dir"] = PLACEHOLDER_MAP["output_dir"]
        updated = True

    return updated, config

def fix_all_configs():
    print(f"🔍 Scanning config folder: {CONFIG_ROOT}")
    for dataset_folder in CONFIG_ROOT.glob("*"):
        for model_folder in dataset_folder.glob("*"):
            for config_file in model_folder.glob("*.json"):
                with open(config_file, "r") as f:
                    config = json.load(f)

                updated, new_config = update_paths(config)
                if updated:
                    with open(config_file, "w") as f:
                        json.dump(new_config, f, indent=2)
                    print(f"✅ Updated: {config_file.relative_to(CONFIG_ROOT)}")
                else:
                    print(f"⏩ Skipped (no change): {config_file.relative_to(CONFIG_ROOT)}")
