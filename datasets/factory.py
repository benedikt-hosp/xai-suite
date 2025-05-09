
# from datasets.robustvision.robustvision_dataset import RobustVisionDataset
from src.utils.robustVision_dataset import RobustVisionDataset
def get_dataset(name: str, params: dict, load_processed= True):
    """
    Factory that returns a Dataset ready for LOOCV.
    """
    if name.lower() in {"robustvision", "rv"}:
        return RobustVisionDataset(
            raw_root             = params["root"],
            processed_root        = params["save_path"],
            seq_len               = params["sequence_length"],
            feature_config   = params["feature_config"],
            load_processed   = load_processed
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
