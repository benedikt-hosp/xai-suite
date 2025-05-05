# utils/dataset_loader.py
from pathlib import Path

from datasets.robustvision.robustvision_dataset import RobustVisionDataset  # ggf. erweitern

# def get_dataset(name, params, skip_load=False):
#     feature_config_path = params.get("feature_config")
#     if feature_config_path:
#         from src.utils.feature_loader import load_feature_config
#         input_features, target_feature, meta_features = load_feature_config(feature_config_path)
#     else:
#         input_features, target_feature, meta_features = None, None, None
#
#     root_path = Path(params["root"]).resolve()
#     print("[DEBUG] Using resolved dataset root:", root_path)
#
#     if name == "robustvision":
#         dataset= RobustVisionDataset(
#             data_dir=root_path,
#             sequence_length=10,
#             input_features=input_features,
#             target_feature=target_feature,
#             meta_features=meta_features
#         )
#
#         # Nur laden, wenn explizit erlaubt und nicht beim Preprocessing
#         if params.get("load_processed", False) and not skip_load:
#             dataset.load_processed_data(params["save_path"])
#
#         return dataset
#
#     raise ValueError(f"Unknown dataset: {name}")




from datasets.robustvision.robustvision_dataset import RobustVisionDataset
from src.utils.dataset_loader_cached import PreprocessedDataset  # NEU

def get_dataset(name, params, skip_load=False):
    feature_config_path = params.get("feature_config")
    if feature_config_path:
        from src.utils.feature_loader import load_feature_config
        input_features, target_feature, meta_features = load_feature_config(feature_config_path)
    else:
        input_features, target_feature, meta_features = None, None, None

    # Verzeichnis auflösen
    root_path = Path(params["root"]).resolve()
    print("[DEBUG] Using resolved dataset root:", root_path)

    # Verwende direkt gecachte Klasse, wenn explizit gewünscht
    if name == "robustvision" and params.get("load_processed", False) and not skip_load:
        print("[INFO] Using PreprocessedDataset")
        return PreprocessedDataset(params["save_path"])

    # Sonst wie bisher
    if name == "robustvision":
        dataset = RobustVisionDataset(
            data_dir=root_path,
            sequence_length=params.get("sequence_length", 10),
            input_features=input_features,
            target_feature=target_feature,
            meta_features=meta_features
        )

        return dataset

    raise ValueError(f"Unknown dataset: {name}")