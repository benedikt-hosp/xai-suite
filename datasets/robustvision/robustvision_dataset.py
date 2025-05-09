import json
import numpy as np
import torch
from pathlib import Path

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, \
    Normalizer, PowerTransformer, FunctionTransformer, Binarizer
from sklearn.utils import resample
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd

# replace with your actual feature‐engineering function
from datasets.robustvision.preprocessor import createFeatures, separate_features_and_targets, global_normalization, \
    subject_wise_normalization

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from datasets.robustvision.preprocessor import createFeatures
from src.utils.data_utils import create_dataloaders_dataset, create_lstm_tensors_dataset


class RobustVisionDataset_FAST(Dataset):
    def __init__(self, raw_root, processed_root, feature_config, seq_len, load_processed=False):

        self.best_transformers = None
        self.raw_root       = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.seq_len        = seq_len
        self.feature_config = feature_config

        cfg = json.loads(Path(feature_config).read_text())
        self.input_features  = cfg["input_features"]
        self.target_feature  = cfg["target_feature"]
        self.feature_names   = list(self.input_features)
        self.batch_size = 460
        self.subject_list = None
        self.transformers = {
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'MaxAbsScaler': MaxAbsScaler,
            'RobustScaler': RobustScaler,
            'QuantileTransformer-Normal': lambda: QuantileTransformer(output_distribution='normal'),
            'QuantileTransformer-Uniform': lambda: QuantileTransformer(output_distribution='uniform'),
            'PowerTransformer-YeoJohnson': lambda: PowerTransformer(method='yeo-johnson'),
            'PowerTransformer-BoxCox': lambda: PowerTransformer(method='box-cox'),
            'Normalizer': Normalizer,
            'Binarizer': lambda threshold=0.0: Binarizer(threshold=threshold),
            'FunctionTransformer-logp1p': lambda func=np.log1p: FunctionTransformer(func),
            'FunctionTransformer-rec': lambda func=np.reciprocal: FunctionTransformer(func),
            'FunctionTransformer-sqrt': lambda func=np.sqrt: FunctionTransformer(func),
        }

        if load_processed and (self.processed_root / "features_raw.pt").exists():
            self._load_cache()
        else:
            self._full_preprocess_and_save()

    def _full_preprocess_and_save(self):
        Xs, ys, subs = [], [], []
        # 1) For each subject, read + feature-engineer + window
        for csv_path in self.raw_root.rglob("*.csv"):

            # 1. read in csv file
            df0 = pd.read_csv(csv_path, sep="\t")

            # 2. remove all non numeric columns
            non_num = df0.select_dtypes(exclude=[np.number]).columns
            df0 = df0.drop(columns=non_num)

            # normalize cols
            df0.columns = (
                df0.columns
                .str.strip()
                .str.replace(r'[\(\)]', '', regex=True)
                .str.replace(r'[ /-]+', '_', regex=True)
                .str.replace(r'__+', '_', regex=True)
                .str.strip('_')
                .str.title()
            )


            df0 = self.clean_data(df0)
            df0 = self.remove_outliers_in_labels(df0, window_size=5, threshold=10)
            df0 = self.detect_and_remove_outliers_in_features_iqr(df0)
            df0 = self.binData(df0, False)


            #
            # df0 = self.clean_data(df0)
            # df0 = self.remove_outliers_in_labels(df0, window_size=5, threshold=10)
            # df0 = self.detect_and_remove_outliers_in_features_iqr(df0)
            # df0 = self.binData(df0, False)

            # feature-engineer (row-wise!)
            df0 = createFeatures(df0, input_features=self.input_features)
            df0["SubjectID"] = csv_path.parent.name

            # pull out numpy arrays for this one subject
            X_sub = df0[self.input_features].to_numpy(dtype=np.float32)
            y_sub = df0[self.target_feature].to_numpy(dtype=np.float32)
            sid = csv_path.parent.name

            # 2) sliding windows *per subject*
            for i in range(len(X_sub) - self.seq_len):
                Xs.append(X_sub[i: i + self.seq_len])
                ys.append(y_sub[i + self.seq_len])
                subs.append(sid)

        # 3) now stack *all* windows
        self.features_raw = torch.from_numpy(np.stack(Xs, axis=0)).float()
        self.targets_raw = torch.from_numpy(np.array(ys, dtype=np.float32)).float()
        self.subject_ids = subs
        self.subject_list = sorted(set(subs))

        # 4) save to disk
        self.processed_root.mkdir(parents=True, exist_ok=True)
        torch.save(self.features_raw, self.processed_root / "features_raw.pt")
        torch.save(self.targets_raw, self.processed_root / "targets_raw.pt")
        with open(self.processed_root / "subjects.json", "w") as f:
            json.dump(self.subject_ids, f)
        with open(self.processed_root / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)

        print(f"Cached {len(self.features_raw)} windows → {self.processed_root}")
    #
    # def prepare_loader_on_df(self, df: pd.DataFrame, batch_size: int, is_train: bool):
    #     # exactly your old logic, but operating on the passed-in df
    #     data = df.copy()
    #     data = self.normalize_data(data)
    #     if is_train:
    #         data = self.calculate_transformations_for_features(data)
    #     else:
    #         data = self.apply_transformations_on_features(data)
    #
    #     data = self.scale_target(data, isTrain=is_train)
    #     sequences = self.create_sequences(data)
    #     features, targets = separate_features_and_targets(sequences)
    #
    #     return create_dataloaders_dataset(
    #         *create_lstm_tensors_dataset(features, targets, is_train),
    #         batch_size=batch_size
    #     )

    def get_fold_from_tensors(self, held_out_subj, batch_size):
        # 1) turn subject_ids into an array so we can boolean‐index
        subs = np.array(self.subject_ids)

        # 2) make masks
        train_mask = subs != held_out_subj
        val_mask = subs == held_out_subj

        # 3) slice your saved tensors
        X_tr = self.features_raw[train_mask]  # → Tensor [N_tr, L, F]
        y_tr = self.targets_raw[train_mask]  # → Tensor [N_tr]
        X_va = self.features_raw[val_mask]  # → Tensor [N_va, L, F]
        y_va = self.targets_raw[val_mask]  # → Tensor [N_va]

        # 4) wrap into TensorDatasets & DataLoaders
        tr_ds = TensorDataset(X_tr, y_tr)
        va_ds = TensorDataset(X_va, y_va)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

        return tr_loader, va_loader

    # 1) your global + subject‐wise normalizer
    def normalize_data(self, data):
        data = global_normalization(data)
        unique_subjects = data['SubjectID'].unique()
        subject_scaler = RobustScaler()
        return subject_wise_normalization(data, unique_subjects, subject_scaler)

    # 2A) train‐only feature‐transforms
    def calculate_transformations_for_features(self, data):
        best = {}
        out = data.copy()
        ideal_skew, ideal_kurt = 0.0, 3.0

        for col in out.columns:
            if col in ("Gt_Depth", "SubjectID"): continue
            best_dist, best_tf = float('inf'), None

            for name, T in self.transformers.items():
                tf = T()
                try:
                    tmp = tf.fit_transform(out[[col]])
                    s = skew(tmp)[0]
                    k = kurtosis(tmp, fisher=False)[0]
                    d = np.hypot(s - ideal_skew, k - ideal_kurt)
                    if d < best_dist:
                        best_dist, best_tf = d, tf
                except ValueError:
                    continue

            self.best_transformers[col] = best_tf
            if best_tf:
                out[col] = best_tf.transform(out[[col]]).squeeze()

        return out

    # 2B) apply those same transformers on val/test
    def apply_transformations_on_features(self, data):
        out = data.copy()
        for col, tf in self.best_transformers.items():
            if tf and col not in ("Gt_Depth", "SubjectID"):
                out[col] = tf.transform(out[[col]]).squeeze()
        return out

    # 3) scale target train/val
    def scale_target(self, data, isTrain=False):
        y = data['Gt_Depth'].values.reshape(-1, 1)
        if isTrain:
            self.target_scaler = StandardScaler() # MinMaxScaler(feature_range=(0, 1000))
            y_s = self.target_scaler.fit_transform(y)
        else:
            y_s = self.target_scaler.transform(y)
        data['Gt_Depth'] = y_s.ravel()
        return data


    # 6.
    def create_sequences(self, df):
        """
        Create sequences of data for time-series analysis.

        :param df: Dataframe with the data to sequence.
        :return: List of sequences, where each sequence is a tuple of (features, target, subject_id).
        """
        sequences = []
        grouped_data = df.groupby('SubjectID')
        for subj_id, group in grouped_data:
            for i in range(len(group) - self.seq_len):
                seq_features = group.iloc[i:i + self.seq_len].drop(columns=['Gt_Depth', 'SubjectID'])
                seq_target = group.iloc[i + self.seq_len]['Gt_Depth']
                sequences.append((seq_features, seq_target, subj_id))
        return sequences

    def clean_data(self, df):
        """
        Perform basic cleaning of the dataframe, including outlier removal and binning.

        :param df: The raw dataframe.
        :return: Cleaned dataframe.
        """
        # print("GT ", target_column_name)
        # df = df.dropna(how='all').replace([np.inf, -np.inf], np.nan).dropna().copy()
        # df = df[(df['Gt_Depth'] > minDepth) & (df['Gt_Depth'] <= maxDepth)]
        # df[target_column_name] = df['Gt_Depth'].multiply(100).reset_index(drop=True)

        # Remove rows where all elements are NaN
        df = df.dropna(how='all')

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().copy()

        # df2 = df[df['Gt_Depth'] > 0.1]
        df2 = df[df['Gt_Depth'] > 0.35]
        df2 = df2[df2['Gt_Depth'] <= 3]

        df2["Gt_Depth"] = df2["Gt_Depth"].multiply(100)

        df2 = df2.reset_index(drop=True)

        return df2

    def detect_and_remove_outliers_in_features_iqr(self, df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1

        mask = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1 * IQR))

        # Filter out the rows with outliers
        new_df = df[~mask.any(axis=1)]
        return new_df

    def remove_outliers_in_labels(self, df, window_size, threshold):
        # Check if 'Gt_Depth' column exists
        if 'Gt_Depth' not in df.columns:
            raise ValueError("Column 'Gt_Depth' not found in the DataFrame")

        # Iterate over the DataFrame
        outlier_indices = []
        for i in range(len(df)):
            # Define the window range
            start = max(i - window_size // 2, 0)
            end = min(i + window_size // 2 + 1, len(df))
            window = df['Gt_Depth'].iloc[start:end]

            # Calculate the median of the window
            mean = np.mean(window)
            # median = np.nanmedian(window)

            # # Check if the current value is an outlier
            if abs(df['Gt_Depth'].iloc[i] - mean) > threshold:
                outlier_indices.append(i)

            # Check if the current value is an outlier
            # if abs(df['Gt_Depth'].iloc[i] - median) > threshold:
            #     outlier_indices.append(i)

        # Check if the outlier indices are in the DataFrame index
        outlier_indices = [idx for idx in outlier_indices if idx in df.index]
        # Now drop the outliers safely
        df_cleaned = df.drop(outlier_indices)
        print(f"Removed {len(outlier_indices)} outlier from data set.")

        return df_cleaned

    def binData(self, df, isGIW=False):
        # Step 1: Bin the target variable
        num_bins = 60  # You can adjust this number
        df['Gt_Depth_bin'] = pd.cut(df['Gt_Depth'], bins=num_bins, labels=False)

        # Step 2: Calculate mean count per bin
        bin_counts = df['Gt_Depth_bin'].value_counts()
        mean_count = bin_counts.mean()

        # Step 3: Resample each bin
        resampled_data = []
        for bin in range(num_bins):
            bin_data = df[df['Gt_Depth_bin'] == bin]
            bin_count = bin_data.shape[0]

            if bin_count == 0:
                continue  # Skip empty bins

            if bin_count < mean_count:
                # Oversample if count is less than mean
                bin_data_resampled = resample(bin_data, replace=True, n_samples=int(mean_count), random_state=123)
            elif bin_count > mean_count:
                # Undersample if count is more than mean
                bin_data_resampled = resample(bin_data, replace=False, n_samples=int(mean_count), random_state=123)
            else:
                # Keep the bin as is if count is equal to mean
                bin_data_resampled = bin_data

            resampled_data.append(bin_data_resampled)

        # Step 4: Combine back into a single DataFrame
        balanced_df = pd.concat(resampled_data)

        if isGIW:
            balanced_df = self.sample_from_bins(balanced_df)
            print(balanced_df['Gt_Depth_bin'].value_counts(normalize=True))

        # Optionally, drop the 'Gt_Depth_bin' column if no longer needed
        balanced_df.drop('Gt_Depth_bin', axis=1, inplace=True)
        return balanced_df

    def sample_from_bins(self, df, fraction=0.12):
        # Ensure that each bin has enough data points for sampling
        sampled_data = df.groupby('Gt_Depth_bin').apply(
            lambda x: x.sample(frac=fraction, random_state=110) if len(x) > 1 else x)

        return sampled_data.reset_index(drop=True)

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
        self.features_raw = self.features_raw[:, :, indices]
        # Update stored feature names
        self.feature_names = [self.feature_names[i] for i in indices]


    def _load_cache(self):
        # load the saved tensors
        self.features_raw = torch.load(self.processed_root / "features_raw.pt").float()
        self.targets_raw  = torch.load(self.processed_root / "targets_raw.pt").float()
        # load subject‐IDs
        with open(self.processed_root / "subjects.json", "r") as f:
            self.subject_ids = json.load(f)
        # feature_names already in self.feature_names
        self.subject_list = sorted(set(self.subject_ids))

    def __len__(self):
        return self.features_raw.shape[0]

    def __getitem__(self, idx):
        # now returns exactly (features, target, subject_id)
        return self.features_raw[idx], self.targets_raw[idx], self.subject_ids[idx]

    def get_fold(self, held_out_subject, batch_size=128):
        # boolean masks over the N windows
        mask_val   = [s == held_out_subject for s in self.subject_ids]
        mask_train = [not m for m in mask_val]

        idx_tr = np.where(mask_train)[0]
        idx_va = np.where(mask_val)[0]

        ds_tr = torch.utils.data.TensorDataset(
            self.features_raw[idx_tr], self.targets_raw[idx_tr]
        )
        ds_va = torch.utils.data.TensorDataset(
            self.features_raw[idx_va], self.targets_raw[idx_va]
        )

        return (
            DataLoader(ds_tr, batch_size=batch_size, shuffle=True),
            DataLoader(ds_va, batch_size=batch_size, shuffle=False),
        )

    def get_fold_(self, held_out_subj, topk=None, mode="keep"):
        # masks
        mask_val = np.array([s == held_out_subj for s in self.subs])
        mask_train = ~mask_val

        Xtr, ytr = self.features_raw[mask_train], self.targets_raw[mask_train]
        Xva, yva = self.features_raw[mask_val], self.targets_raw[mask_val]

        # 1) fit scaler on train, apply to train+val
        #    flatten (N*seq_len, F) to fit per-feature
        ns, sl, nf = Xtr.shape
        scaler = StandardScaler().fit(
            Xtr.reshape(-1, nf).numpy()
        )

        def apply(X):
            flat = scaler.transform(X.reshape(-1, nf).numpy())
            return torch.from_numpy(flat.reshape(-1, sl, nf).astype(np.float32))

        Xtr_s = apply(Xtr)
        Xva_s = apply(Xva)

        # 2) optionally drop/select top-K features based on some ranking you computed
        if topk is not None:
            # e.g. load your ranking list
            ranks = json.loads((self.root / f"ranking_{held_out_subj}.json").read_text())
            keep = [r["feature"] for r in ranks[:topk]]
            idxs = [self.features.index(f) for f in keep]
            Xtr_s = Xtr_s[:, :, idxs]
            Xva_s = Xva_s[:, :, idxs]

        # 3) build loaders
        ds_tr = TensorDataset(Xtr_s, ytr)
        ds_va = TensorDataset(Xva_s, yva)
        loader_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True)
        loader_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False)
        return loader_tr, loader_va


