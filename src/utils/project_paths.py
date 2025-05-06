# Common folders
from pathlib import Path

# Projekt-Wurzel (anpassen, falls notwendig)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

print(PROJECT_ROOT)

# Zentrale Verzeichnisse
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "robustvision"
DATA_PROCESSED = DATA_DIR  / "processed" / "robustvision"
FEATURE_CONFIG = PROJECT_ROOT / "datasets" / "robustvision" / "features.json"
EXPERIMENT_RESULTS = PROJECT_ROOT / "results"
FEATURE_RANKING_LISTS = PROJECT_ROOT / "results" / "feature_rankings_lists"
FEATURE_EVAL_TASKS = PROJECT_ROOT / "results" / "feature_evaluation_tasks"
FEATURE_EVAL_PERFORMANCE_MEASURES = PROJECT_ROOT / "results" / "feature_evaluation_performance_measures"


CONFIG_ROOT = PROJECT_ROOT / "experiments" / "configs"


from pathlib import Path

def resolve_paths(cfg, dataset_name=None, model_name=None, method_name=None):
    replacements = {
        "__DATA_RAW__": str(DATA_RAW),
        "__DATA_PROCESSED__": str(DATA_PROCESSED),
        "__FEATURE_CONFIG__": str(FEATURE_CONFIG),
        "__RESULTS_DIR__": str(EXPERIMENT_RESULTS),
        "__FEATURE_RANKING_LISTS__": str(FEATURE_RANKING_LISTS),
    }

    def replace_placeholders(value):
        if isinstance(value, str):
            for placeholder, real_path in replacements.items():
                value = value.replace(placeholder, real_path)
        return value

    # Gehe rekursiv durch das Dictionary
    def resolve_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                resolve_dict(v)
            elif isinstance(v, list):
                d[k] = [replace_placeholders(x) for x in v]
            else:
                d[k] = replace_placeholders(v)

    resolve_dict(cfg)

    # Optional: Dynamischer Output-Pfad
    if "eval" in cfg and "output_dir" in cfg["eval"] and dataset_name and model_name and method_name:
        cfg["eval"]["output_dir"] = str(
            Path(EXPERIMENT_RESULTS) / f"{dataset_name}_{model_name}_{method_name}"
        )

    return cfg
