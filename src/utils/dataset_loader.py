from pathlib import Path
# from datasets.robustvision.robustvision_dataset import RobustVisionDataset

def get_dataset(name: str, params: dict):  # -> RobustVisionDataset:
    pass
#     """
#     Returns a dataset instance. Use `load_processed` to switch between raw load vs. cache.
#     """
#     if name.lower() in ("robustvision", "rv"):
#         return RobustVisionDataset(
#             raw_data_dir   = Path(params["root"]),
#             processed_data_dir = Path(params["save_path"]),
#             sequence_length   = params["sequence_length"],
#             input_features    = params.get("input_features", []),
#             target_feature    = params.get("target_feature", "Gt_Depth"),
#             use_preprocessed  = params.get("load_processed", False),
#         )
#     raise ValueError(f"Unknown dataset: {name}")