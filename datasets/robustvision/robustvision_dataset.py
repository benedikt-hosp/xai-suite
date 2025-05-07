import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd

# replace with your actual feature‐engineering function
from datasets.robustvision.preprocessor import createFeatures


class RobustVisionDataset(Dataset):
    """
    Two-stage dataset for RobustVision:
     1) preprocess once: read raw CSVs → engineer features → build sliding windows → save tensors
     2) on‐the‐fly: load saved tensors → slice out one subject → return DataLoaders for LOOCV
    """

    def __init__(
        self,
        raw_root: str,
        processed_root: str,
        feature_config: str,
        seq_len: int,
        load_processed: bool = False,
    ):
        self.raw_root       = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.seq_len        = seq_len

        cfg = json.loads(Path(feature_config).read_text())
        self.input_features = cfg["input_features"]
        self.target_feature = cfg["target_feature"]

        if load_processed and (self.processed_root / "features.pt").exists():
            self._load_cache()
        else:
            self._full_preprocess_and_save()

    def _read_all_csv(self) -> pd.DataFrame:
        dfs = []
        for csv_path in self.raw_root.rglob("*.csv"):
            df0 = pd.read_csv(csv_path, sep="\t")
            df0["SubjectID"] = csv_path.parent.name
            dfs.append(df0)
        df = pd.concat(dfs, ignore_index=True)
        # normalize column names
        df.columns = (
            df.columns
              .str.strip()
              .str.replace(r'[\(\)]', '', regex=True)
              .str.replace(r'[ /-]+', '_', regex=True)
              .str.replace(r'__+', '_', regex=True)
              .str.strip('_')
              .str.title()
        )
        df.rename(columns=lambda c: "SubjectID" if c.lower()=="subjectid" else c, inplace=True)
        return df

    def _full_preprocess_and_save(self):
        df = self._read_all_csv()

        # 1) feature engineering
        df = createFeatures(df, input_features=self.input_features)

        # 2) sliding‐window sequences per subject
        X, y, subs = [], [], []
        arr_X   = df[self.input_features].astype(np.float32).to_numpy()
        arr_y   = df[self.target_feature].astype(np.float32).to_numpy()
        arr_sub = df["SubjectID"].to_numpy()

        for subj in np.unique(arr_sub):
            idxs = np.where(arr_sub == subj)[0]
            X_sub, y_sub = arr_X[idxs], arr_y[idxs]
            for i in range(len(X_sub) - self.seq_len):
                X.append(X_sub[i : i + self.seq_len])
                y.append(y_sub[i + self.seq_len])
                subs.append(subj)

        X = np.stack(X, axis=0)    # shape (N, seq_len, F)
        y = np.array(y, dtype=np.float32)
        subs = np.array(subs)

        # 3) save to disk
        self.processed_root.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(X),        self.processed_root / "features.pt")
        torch.save(torch.from_numpy(y),        self.processed_root / "targets.pt")
        torch.save(subs,                       self.processed_root / "subjects.pt")
        with open(self.processed_root / "feature_names.json", "w") as f:
            json.dump(self.input_features, f)

        # cache in memory
        self.features      = torch.from_numpy(X)
        self.targets       = torch.from_numpy(y)
        self.subjects      = subs.tolist()
        self.feature_names = list(self.input_features)

        # after you set `self.features`, `self.targets` and `self.subjects` …
        self.features_tensor = self.features
        self.targets_tensor = self.targets

    def _load_cache(self):
        # torch.load in PyTorch ≥2.6 needs weights_only=False to load generic tensors
        self.features      = torch.load(self.processed_root / "features.pt", weights_only=False).float()
        self.targets       = torch.load(self.processed_root / "targets.pt",  weights_only=False).float()
        subs               = torch.load(self.processed_root / "subjects.pt", weights_only=False)
        self.subjects      = subs.tolist()
        self.feature_names = json.load(open(self.processed_root / "feature_names.json"))

        # after you set `self.features`, `self.targets` and `self.subjects` …
        self.features_tensor = self.features
        self.targets_tensor = self.targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def set_selected_features(self, selected_features, mode="keep"):
        """
        Subset your feature tensor and names according to `mode`:
        - "keep": keep only `selected_features`
        - "remove": drop `selected_features`
        """
        if mode not in ("keep", "remove"):
            raise ValueError("mode must be 'keep' or 'remove'")

        # Determine new feature indices
        if mode == "keep":
            indices = [self.feature_names.index(f) for f in selected_features]
        else:  # remove
            indices = [i for i, f in enumerate(self.feature_names) if f not in selected_features]

        # Slice the features tensor (N, seq_len, F) to only those indices
        self.features_tensor = self.features_tensor[:, :, indices]
        # Update stored feature names
        self.feature_names = [self.feature_names[i] for i in indices]


    def get_fold_loaders(self, held_out_subject: str, batch_size: int = 128):
        """
        Build train/val loaders for leave‐one‐subject‐out CV.
        """
        mask_val   = [s == held_out_subject for s in self.subjects]
        mask_train = [not m for m in mask_val]

        idx_tr = np.where(mask_train)[0]
        idx_va = np.where(mask_val)[0]

        ds_tr = TensorDataset(self.features[idx_tr], self.targets[idx_tr])
        ds_va = TensorDataset(self.features[idx_va], self.targets[idx_va])

        loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
        return loader_tr, loader_va

# import json
# import pickle
# from pathlib import Path
#
# import torch
# import numpy as np
# import pandas as pd
# import os
# from scipy.stats import skew, kurtosis
# from sklearn.preprocessing import (
#     PowerTransformer, Normalizer, MaxAbsScaler, RobustScaler,
#     QuantileTransformer, StandardScaler, MinMaxScaler,
#     FunctionTransformer, Binarizer
# )
# import warnings
#
# from datasets.base.abstract_dataset import AbstractDatasetClass
# from datasets.robustvision.preprocessor import clean_data, detect_and_remove_outliers_in_features_iqr, \
#     remove_outliers_in_labels, binData, createFeatures, global_normalization, subject_wise_normalization, \
#     separate_features_and_targets
# from src.utils.data_utils import create_lstm_tensors_dataset, create_dataloaders_dataset
# from src.utils.project_paths import DATA_PROCESSED
#
# warnings.filterwarnings("ignore")
# pd.set_option('display.max_columns', None)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import os
# import torch
# import pandas as pd
# import pickle
# import json
# from pathlib import Path
# from torch.utils.data import Dataset, DataLoader, Subset
# import numpy as np
# from scipy.stats import skew, kurtosis
# from sklearn.preprocessing import (
#     PowerTransformer, Normalizer, MaxAbsScaler, RobustScaler,
#     QuantileTransformer, StandardScaler, MinMaxScaler,
#     FunctionTransformer
# )
# from datasets.robustvision.preprocessor import (
#     clean_data, detect_and_remove_outliers_in_features_iqr,
#     remove_outliers_in_labels, binData, createFeatures,
#     global_normalization, subject_wise_normalization
# )
#
# class RobustVisionDataset(Dataset):
#     """
#     Dataset for RobustVision: supports one-time preprocessing and caching of tensors,
#     then fast DataLoader creation per subject for LOOCV.
#     """
#     def __init__(
#         self,
#         root: str,
#         save_path: str,
#         sequence_length: int,
#         feature_config: str,
#         split: str = "all",
#         load_processed: bool = False
#     ):
#         self.feature_names = None
#         self.subject_list = None
#         self.targets_tensor = None
#         self.features_tensor = None
#         self.raw_data_dir = Path(root)
#         self.processed_data_dir = Path(save_path)
#         self.sequence_length = sequence_length
#         self.split = split
#         self.batch_size = 460
#
#         # Load feature/target and scaler config
#         with open(feature_config) as f:
#             feat_cfg = json.load(f)
#
#         self.meta_features = feat_cfg["meta_features"]  # ← e.g. ["SubjectID"]
#         self.target_feature = feat_cfg["target_feature"]  # ← "Gt_Depth"
#         self.input_features = feat_cfg["input_features"]  # ← your 40-odd names
#         self.scaler_config = feat_cfg.get("scaler_config", {})
#
#         self.scaler_config = feat_cfg.get("scaler_config", {})
#
#         # Transformer candidates for skew/kurtosis
#         self.transformers = {
#             "standard": StandardScaler,
#             "minmax": MinMaxScaler,
#             "robust": RobustScaler,
#             "quantile": QuantileTransformer,
#             "power": PowerTransformer,
#             "maxabs": MaxAbsScaler,
#             "normalizer": Normalizer,
#             "identity": FunctionTransformer
#         }
#
#         if load_processed and (self.processed_data_dir / 'features.pt').exists():
#             self.load_processed_data()
#         else:
#             # Prepare cleaned DataFrame for on-the-fly loaders
#             df = self._read_raw()
#             # print("⎯⎯⎯ RAW COLUMNS ⎯⎯⎯")
#             # print(df.columns.tolist())
#             # print("⎯⎯⎯––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")
#
#             df = self.create_features(df)
#             # df = self.normalize_data(df)
#             self.cleaned_df = df
#             self.subject_ids = df["SubjectID"].tolist()
#             self.subject_list = sorted(set(self.subject_ids))
#
#
#
#     def _read_raw(self) -> pd.DataFrame:
#         records = []
#         for path in self.raw_data_dir.rglob('*.csv'):
#             # assume each CSV lives in a folder named e.g. ".../P01/data.csv"
#             subject = path.parent.name
#             # if tab-separated:
#             df0 = pd.read_csv(path, sep='\t')
#             df0['SubjectID'] = subject
#             records.append(df0)
#
#         df = pd.concat(records, ignore_index=True)
#
#         # normalize column names just once:
#         df.columns = (
#             df.columns
#             .str.strip()
#             .str.replace(r'[\(\)]', '', regex=True)
#             .str.replace(r'[ /-]+', '_', regex=True)
#             .str.replace(r'__+', '_', regex=True)
#             .str.strip('_')
#             .str.title()
#         )
#
#         # force the meta‐feature name to match exactly what your config wants
#         if 'Subjectid' in df.columns:
#             df.rename(columns={'Subjectid': 'SubjectID'}, inplace=True)
#
#
#         return df
#
#
#     def preprocess_and_save(self):
#         # 1) Read raw
#         df = self._read_raw()
#         # 2) Feature engineering
#         df = createFeatures(df, input_features=self.input_features)
#         # 3) Keep only needed cols
#         # keep = self.meta_features + [self.target_feature] + self.input_features
#         # df = df[keep]
#
#         # 4) Normalize numeric
#         # df = self.normalize_data(df)
#
#         # 5) Transform + scale
#         # df = self.calculate_transformations_for_features(df)
#         # df = self.scale_target(df, isTrain=True)
#
#         # 6) Create sequences
#         feats, targets, subjects = self.create_sequences(df)
#         # 7) Convert to tensors
#         self.features_tensor = torch.tensor(feats, dtype=torch.float32)
#         self.targets_tensor  = torch.tensor(targets, dtype=torch.float32)
#         # self.subjects_tensor = torch.tensor(subs)
#         self.feature_names   = self.input_features.copy()
#
#         # 8) Save all
#         self.processed_data_dir.mkdir(parents=True, exist_ok=True)
#         torch.save(self.features_tensor, self.processed_data_dir / "features.pt")
#         torch.save(self.targets_tensor,  self.processed_data_dir / "targets.pt")
#         torch.save(subjects, self.processed_data_dir / "subjects.pt")
#
#         with open(self.processed_data_dir / "feature_names.json", "w") as f:
#             json.dump(self.feature_names, f)
#         with open(self.processed_data_dir / "scalers.pkl", "wb") as f:
#             pickle.dump({
#                 'feature_transformers': getattr(self, 'feature_transformers', {}),
#                 'target_scaler': getattr(self, 'target_scaler', None)
#             }, f)
#
#         # Prepare splits
#         self.subject_ids  = subjects.tolist()
#         self.subject_list = sorted(set(self.subject_ids))
#     #
#     # def preprocess_and_save(self):
#     #     # 1) Read & normalize column names
#     #     df = self._read_raw()
#     #
#     #     # 2) Fix that lower-case 'Subjectid' → 'SubjectID'
#     #     df = df.rename(columns={c: "SubjectID"
#     #                             for c in df.columns
#     #                             if c.lower() == "subjectid"})
#     #
#     #     # 3) Create all of your derived features
#     #     df = self.create_features(df)
#     #
#     #     # 4) Keep *only* the columns you care about:
#     #     #    meta_features + target_feature + input_features
#     #     keep = self.meta_features + [self.target_feature] + self.input_features
#     #     missing = set(keep) - set(df.columns)
#     #     if missing:
#     #         raise KeyError(f"Columns missing after feature‐gen: {missing}")
#     #     df = df[keep]
#     #
#     #     # 5) Normalize *only* your numeric inputs
#     #     df = self.normalize_data(df)
#     #
#     #     # after you’ve filtered and normalized `ds.input_data`, but *before* you drop SubjectID:
#     #     subjects = torch.tensor(df.input_data["SubjectID"].values, dtype=torch.long)
#     #     # now drop the column so features_tensor is purely numeric
#     #     feat_df = df.input_data.drop(columns=["SubjectID"])
#     #
#     #
#     #     # 6) Fit+apply your skew/kurtosis transforms
#     #     df = self.calculate_transformations_for_features(df)
#     #
#     #     # 7) Scale the target
#     #     df = self.scale_target(df, isTrain=True)
#     #
#     #     # 8) Sequence + tensorize + dump everything
#     #     feats, targets, subjects = self.create_sequences(df)
#     #     # convert to feature/target tensors
#     #     features_tensor = torch.tensor(feat_df[df.input_features].values, dtype=torch.float32)
#     #     targets_tensor = torch.tensor(df.input_data[df.target_feature].values, dtype=torch.float32)
#     #
#     #     # save them all
#     #     torch.save(features_tensor, self.processed_data_dir  / "features.pt")
#     #     torch.save(targets_tensor, self.processed_data_dir  / "targets.pt")
#     #     torch.save(subjects, self.processed_data_dir  / "subjects.pt")
#     #
#     #     self.subject_list = subjects.tolist()
#     #     # self.feature_names = self.input_features
#     #
#     #     # # save to disk
#     #     # self.processed_data_dir.mkdir(exist_ok=True, parents=True)
#     #     # torch.save(self.features_tensor, self.processed_data_dir / "features.pt")
#     #     # torch.save(self.targets_tensor, self.processed_data_dir / "targets.pt")
#     #     #
#     #     with open(self.processed_data_dir / "subjects.json", "w") as f:
#     #         json.dump(self.subject_list, f)
#     #     with open(self.processed_data_dir / "feature_names.json", "w") as f:
#     #         json.dump(self.feature_names, f)
#     #     with open(self.processed_data_dir / "scalers.pkl", "wb") as f:
#     #         pickle.dump({"feature_transformers": self.feature_transformers,
#     #                      "target_scaler": self.target_scaler}, f)
#     def __len__(self):
#         return self.features_tensor.size(0)
#
#     def __getitem__(self, idx):
#         return self.features_tensor[idx], self.targets_tensor[idx]
#
#     def get_data_loader(self, train_index, val_index=None, batch_size=100):
#         train_loader = self._make_loader(train_index, batch_size, shuffle=True)
#         val_loader = (
#             self._make_loader(val_index, batch_size, shuffle=False)
#             if val_index is not None else None
#         )
#         return train_loader, val_loader, self.features_tensor.size(2)
#
#     def _make_loader(self, subject_index, batch_size, shuffle=False):
#         subjects = subject_index if isinstance(subject_index, list) else [subject_index]
#         idxs = [i for i, s in enumerate(self.subject_list) if s in subjects]
#         if not idxs:
#             raise ValueError(f"No data for subjects {subjects}")
#         subset = Subset(self, idxs)
#         return DataLoader(
#             subset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=8,
#             pin_memory=True,
#             persistent_workers=True
#         )
#
#     def get_fold_loaders(self, held_out_subject, batch_size=None):
#         """
#         Leave-one-subject-out: hold out all windows of one subject.
#         """
#         if batch_size is None:
#             batch_size = self.batch_size
#
#         mask_val = [sid == held_out_subject for sid in self.subject_ids]
#         mask_train = [not m for m in mask_val]
#
#         X, y = self.features_tensor, self.targets_tensor
#         train_ds = Subset(torch.utils.data.TensorDataset(X, y), np.where(mask_train)[0])
#         val_ds = Subset(torch.utils.data.TensorDataset(X, y), np.where(mask_val)[0])
#
#         train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
#         return train_loader, val_loader
#
#     def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Generate new features (e.g., gaze ratios, magnitudes) from raw.
#         """
#         return createFeatures(df, input_features=self.input_features)
#     #
#     # def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
#     #     # pull out just your inputs
#     #     X = df[self.input_features].values
#     #     X_scaled = RobustScaler().fit_transform(X)
#     #
#     #     # rebuild the frame
#     #     out = pd.DataFrame(X_scaled, columns=self.input_features, index=df.index)
#     #     out[self.target_feature] = df[self.target_feature].values
#     #     for m in self.meta_features:  # now meta_features is a real list
#     #         out[m] = df[m].values
#     #     return out
#     # #
#     # def calculate_transformations_for_features(self, df: pd.DataFrame) -> pd.DataFrame:
#     #     """
#     #     Fit best transformer per feature to approach normality and store.
#     #     """
#     #     self.feature_transformers = {}
#     #     transformed = df.copy()
#     #     for col in df.columns:
#     #         if col in [self.target_feature, 'SubjectID']:
#     #             continue
#     #         best = None
#     #         best_name = None
#     #         best_dist = float('inf')
#     #         for name, cls in self.transformers.items():
#     #             tf = cls()
#     #             try:
#     #                 tmp = tf.fit_transform(df[[col]])
#     #                 dist = np.sqrt((skew(tmp)[0])**2 + (kurtosis(tmp, fisher=False)[0] - 3)**2)
#     #                 if dist < best_dist:
#     #                     best_dist = dist; best = tf; best_name = name
#     #             except Exception:
#     #                 continue
#     #         if best is not None:
#     #             transformed[col] = best.transform(df[[col]]).squeeze()
#     #             self.feature_transformers[col] = best
#     #     return transformed
#     #
#     # def apply_transformations_on_features(self, df: pd.DataFrame) -> pd.DataFrame:
#     #     out = df.copy()
#     #     for col, tf in self.feature_transformers.items():
#     #         out[col] = tf.transform(df[[col]]).squeeze()
#     #     return out
#     #
#     # def scale_target(self, df: pd.DataFrame, isTrain: bool) -> pd.DataFrame:
#     #     y = df[self.target_feature].values.reshape(-1,1)
#     #     if isTrain:
#     #         self.target_scaler = self._select_scaler(self.scaler_config)
#     #         if self.target_scaler:
#     #             y = self.target_scaler.fit_transform(y)
#     #     else:
#     #         if self.target_scaler:
#     #             y = self.target_scaler.transform(y)
#     #     df[self.target_feature] = y.ravel()
#     #     return df
#
#     def _select_scaler(self, cfg: dict):
#         if cfg.get('use_minmax'): return MinMaxScaler(feature_range=(0,1000))
#         if cfg.get('use_standard_scaler'): return StandardScaler()
#         if cfg.get('use_robust_scaler'): return RobustScaler()
#         if cfg.get('use_quantile_transformer'): return QuantileTransformer(output_distribution='normal')
#         if cfg.get('use_power_transformer'): return PowerTransformer()
#         if cfg.get('use_max_abs_scaler'): return MaxAbsScaler()
#         return None
#
#     # def create_sequences(self, df: pd.DataFrame):
#     #     seqs, targets, subs = [], [], []
#     #     for subj, grp in df.groupby('SubjectID'):
#     #         arr = grp.to_numpy()
#     #         for i in range(len(arr) - self.sequence_length):
#     #             seqs.append(arr[i:i+self.sequence_length, :][..., [df.columns.get_loc(f) for f in self.input_features]])
#     #             targets.append(arr[i+self.sequence_length, df.columns.get_loc(self.target_feature)])
#     #             subs.append(subj)
#     #     return np.array(seqs), np.array(targets), np.array(subs)
#
#     def create_sequences(self, df: pd.DataFrame):
#         """
#         Build (N, seq_len, F) feature tensors and (N,) target vector grouped by SubjectID.
#         """
#         seqs, targets, subs = [], [], []
#
#         # 1) grab the pure numeric feature‐matrix and target‐vector
#         X_all = df[self.input_features].astype(np.float32).to_numpy()
#         y_all = df[self.target_feature].astype(np.float32).to_numpy()
#         subj_all = df['SubjectID'].to_numpy()
#
#         # 2) for each subject, roll a sliding window
#         for subj in np.unique(subj_all):
#             idxs = np.where(subj_all == subj)[0]
#             X_sub = X_all[idxs]
#             y_sub = y_all[idxs]
#             for i in range(len(X_sub) - self.sequence_length):
#                 seqs.append(X_sub[i: i + self.sequence_length])
#                 targets.append(y_sub[i + self.sequence_length])
#                 subs.append(subj)
#
#         # 3) stack into real float32 arrays
#         return (
#             np.stack(seqs, axis=0),  # shape (num_seqs, seq_len, n_features)
#             np.array(targets, dtype=np.float32),
#             np.array(subs)
#         )
#
#     def set_selected_features(self, selected_features, mode="keep"):
#         """
#         Subset your feature tensor and names according to `mode`:
#         - "keep": keep only `selected_features`
#         - "remove": drop `selected_features`
#         """
#         if mode not in ("keep", "remove"):
#             raise ValueError("mode must be 'keep' or 'remove'")
#
#         # Determine new feature indices
#         if mode == "keep":
#             indices = [self.feature_names.index(f) for f in selected_features]
#         else:  # remove
#             indices = [i for i, f in enumerate(self.feature_names) if f not in selected_features]
#
#         # Slice the features tensor (N, seq_len, F) to only those indices
#         self.features_tensor = self.features_tensor[:, :, indices]
#         # Update stored feature names
#         self.feature_names = [self.feature_names[i] for i in indices]
#
#     def save_processed_data(self, param):
#         pass
#
#     def load_processed_data(self):
#         # 1) load the feature & target tensors
#         self.features_tensor = torch.load(self.processed_data_dir / 'features.pt')
#         self.targets_tensor  = torch.load(self.processed_data_dir / 'targets.pt')
#
#         self.subject_list = json.load(open(self.processed_data_dir / "subjects.json"))
#         self.feature_names = json.load(open(self.processed_data_dir / "feature_names.json"))
#
#         # # 2) load subjects list from JSON, not with torch.load
#         # with open(self.processed_data_dir / 'subjects.json', 'r') as f:
#         #     subjects = json.load(f)
#         #
#         # # 3) feature names are also JSON
#         # with open(self.processed_data_dir / 'feature_names.json', 'r') as f:
#         #     self.feature_names = json.load(f)
#         #
#         # # 4) scalers remain a pickle
#         # with open(self.processed_data_dir / 'scalers.pkl', 'rb') as f:
#         #     scalers = pickle.load(f)
#         # self.feature_transformers = scalers.get('feature_transformers', {})
#         # self.target_scaler       = scalers.get('target_scaler', None)
#         #
#         # # 5) rebuild your subject IDs and list
#         # #    (they were strings, so no need for a tensor)
#         # self.subject_ids  = subjects
#         # self.subject_list = sorted(set(self.subject_ids))
#
#     # def load_processed_data(self):
#     #     self.features_tensor = torch.load(self.processed_data_dir / 'features.pt')
#     #     self.targets_tensor = torch.load(self.processed_data_dir / 'targets.pt')
#     #     subjects = torch.load(self.processed_data_dir / 'subjects.json')
#     #     self.feature_names = json.load(open(self.processed_data_dir / 'feature_names.json'))
#     #     scalers = pickle.load(open(self.processed_data_dir / 'scalers.pkl', 'rb'))
#     #     self.feature_transformers = scalers.get('feature_transformers', {})
#     #     self.target_scaler = scalers.get('target_scaler', None)
#     #
#     #     self.subject_ids = subjects.tolist()
#     #     self.subject_list = sorted(set(self.subject_ids))
#
#
#
# def get_dataset(name: str, params: dict) -> RobustVisionDataset:
#     if name.lower() in ['rv','robustvision']:
#         return RobustVisionDataset(
#             root=params['root'],
#             save_path=params['save_path'],
#             sequence_length=params['sequence_length'],
#             feature_config=params['feature_config'],
#             split=params.get('split','all'),
#             load_processed=params.get('load_processed',False)
#         )
#     raise ValueError(f"Unknown dataset: {name}")
