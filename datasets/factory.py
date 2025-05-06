
from datasets.robustvision.robustvision_dataset import RobustVisionDataset

def get_dataset(name: str, params: dict, skip_load: bool = False):
    """
    Factory that returns a Dataset ready for LOOCV.
    """
    if name.lower() in {"robustvision", "rv"}:
        return RobustVisionDataset(
            root             = params["root"],
            save_path        = params["save_path"],
            sequence_length  = params["sequence_length"],
            feature_config   = params["feature_config"],
            split            = params.get("split", "all"),
            load_processed   = params.get("load_processed", False)
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
