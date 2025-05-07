# src/utils/evaluate_loocv.py

import logging

import torch
from torch.utils.data import Subset, DataLoader, TensorDataset

from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import logging

logger = logging.getLogger(__name__)
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler
import logging

from sklearn.preprocessing import RobustScaler, StandardScaler


def run_loocv(trainer_factory, dataset):
    X_raw = dataset.features_raw.numpy()  # shape (N, seq_len, F)
    y_raw = dataset.targets_raw.numpy()  # shape (N,)
    subs = np.array(dataset.subject_ids)  # shape (N,)

    fold_maes = []
    for fold_idx, subj in enumerate(dataset.subject_list, 1):
        # boolean masks
        train_mask = subs != subj
        val_mask = subs == subj

        X_tr, y_tr = X_raw[train_mask], y_raw[train_mask]
        X_va, y_va = X_raw[val_mask], y_raw[val_mask]

        # ---- Fit feature scaler on train only ----
        Ntr, L, F = X_tr.shape
        feat_scaler = RobustScaler().fit(X_tr.reshape(-1, F))
        X_tr_scaled = feat_scaler.transform(X_tr.reshape(-1, F)).reshape(Ntr, L, F)
        X_va_scaled = feat_scaler.transform(X_va.reshape(-1, F)).reshape(-1, L, F)

        # ---- Fit target scaler on train only (optional) ----
        tgt_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))
        y_tr_scaled = tgt_scaler.transform(y_tr.reshape(-1, 1)).ravel()
        # we'll evaluate MAE on the *original* scale:

        # ---- Build dataloaders ----
        tr_loader = DataLoader(TensorDataset(
            torch.from_numpy(X_tr_scaled), torch.from_numpy(y_tr_scaled)),
            batch_size=460, shuffle=True)
        va_loader = DataLoader(TensorDataset(
            torch.from_numpy(X_va_scaled), torch.from_numpy(y_va)),
            batch_size=460, shuffle=False)

        # ---- Train & evaluate ----
        trainer = trainer_factory(input_dim=F, feature_names=dataset.feature_names)
        trainer.set_dataloaders(tr_loader, va_loader)
        trainer.set_features(dataset.feature_names)
        mae_scaled = trainer.run_fold(fold_idx)

        # inverse‐transform the MAE back to original scale:
        # since MAE is linear, we can multiply by tgt_scaler.scale_[0]
        mae_orig = mae_scaled * tgt_scaler.scale_[0]
        fold_maes.append(mae_orig)

    return dataset.subject_list, fold_maes, np.mean(fold_maes)

#
# logger = logging.getLogger(__name__)
#
# def run_loocv(trainer_factory, dataset):
#     X = dataset.features_tensor.numpy()    # (N, T, F)
#     y = dataset.targets_tensor.numpy()     # (N,)
#     subs = np.array(dataset.subjects)  # length = N, per-sample subject IDs
#
#     fold_maes = []
#     unique_subs = np.unique(subs)
#     for fold_idx, held in enumerate(unique_subs, 1):
#         logger.info(f"[LOOCV] Fold {fold_idx}/{len(unique_subs)}: held-out subject='{held}'")
#
#         # 1) split train vs val by subject
#         train_mask = (subs != held)
#         val_mask   = (subs == held)
#         X_tr, y_tr = X[train_mask], y[train_mask]
#         X_vl, y_vl = X[val_mask],   y[val_mask]
#
#         # 2) fit feature scaler on TRAIN only
#         Ntr, T, F = X_tr.shape
#         feat_s     = RobustScaler().fit(X_tr.reshape(-1, F))
#         X_tr_s     = feat_s.transform(X_tr.reshape(-1, F)).reshape(Ntr, T, F)
#         X_vl_s     = feat_s.transform(X_vl.reshape(-1, F)).reshape(-1, T, F)
#
#         # 3) fit target scaler on TRAIN only
#         tgt_s  = StandardScaler().fit(y_tr.reshape(-1,1))
#         y_tr_s = tgt_s.transform(y_tr.reshape(-1,1)).ravel()
#         y_vl_s = tgt_s.transform(y_vl.reshape(-1,1)).ravel()
#
#         # 4) build PyTorch loaders
#         train_ds = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_s))
#         val_ds   = TensorDataset(torch.from_numpy(X_vl_s), torch.from_numpy(y_vl_s))
#         train_ld = DataLoader(train_ds, batch_size=460, shuffle=True)
#         val_ld   = DataLoader(val_ds,   batch_size=460, shuffle=False)
#
#         # 5) train & evaluate
#         trainer = trainer_factory(F)
#         trainer.set_dataloaders(train_ld, val_ld)
#         trainer.setup()
#         mae     = trainer.run_fold(fold_idx)
#         fold_maes.append(mae)
#
#     mean_mae = float(np.mean(fold_maes))
#     return fold_maes, mean_mae



# def run_loocv(trainer_factory, dataset, batch_size=32):
#     """
#     trainer_factory: fn(input_dim, feature_names) -> TorchTrainer
#     dataset: must have:
#       - dataset.subject_list: list of unique subject IDs (strings)
#       - dataset.subject_ids: array-like of length N, giving subject ID for each sample
#       - dataset.input_size: int
#       - dataset.feature_names: list[str]
#     """
#     subject_results = []
#     subject_list = dataset.subject_list
#     n_folds = len(subject_list)
#     logger.info(f"[LOOCV] Starting {n_folds}-fold LOOCV")
#
#     for fold_idx, subj in enumerate(subject_list, start=1):
#         logger.info(f"[LOOCV] Fold {fold_idx}/{n_folds}: held-out subject={subj!r}")
#
#         # build train/val index lists
#         all_ids = dataset.subject_ids  # e.g. numpy array of length N
#         train_idx = [i for i, sid in enumerate(all_ids) if sid != subj]
#         val_idx   = [i for i, sid in enumerate(all_ids) if sid == subj]
#         logger.debug(f"[LOOCV]  → |train|={len(train_idx)}  |val|={len(val_idx)}")
#
#         # wrap in Subset + DataLoader
#         train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
#         val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False)
#
#         # new trainer
#         trainer = trainer_factory(len(dataset.feature_names)) # dataset.input_size, dataset.feature_names)
#         trainer.set_dataloaders(train_loader, val_loader)
#
#         # run the fold
#         mae = trainer.run_fold(fold_idx)
#         logger.info(f"[LOOCV]  → Fold MAE={mae:.4f}")
#         subject_results.append((subj, mae))
#
#     mean_mae = sum(m for _, m in subject_results) / n_folds
#     logger.info(f"[LOOCV] Done. mean_MAE={mean_mae:.4f}")
#     return subject_results, mean_mae



# import torch
# def run_loocv(trainer_factory, dataset):
#     """
#     Leave-one-subject-out cross-validation loop.
#     trainer_factory:  zero-arg callable that builds a fresh Trainer
#     dataset:          your RobustVisionDataset instance
#     """
#     subject_scores = {}
#     for subject in dataset.subject_list:
#         # build new loaders for this fold
#         train_loader, val_loader, input_dim = dataset.get_data_loader(
#             train_index=[s for s in dataset.subject_list if s != subject],
#             val_index=[subject],
#             batch_size=dataset.batch_size  # assume you stored batch_size in dataset
#         )
#
#         # new trainer—for clean weights/scheduler each fold
#         # trainer = trainer_factory(input_dim)
#         # # train & validate
#         # mae = trainer.run_fold(train_loader, val_loader)
#         # subject_scores[subject] = mae
#
#         trainer = trainer_factory(input_dim)
#         trainer.set_features(dataset.input_features)
#         trainer.set_dataloaders(train_loader, val_loader)
#         trainer.setup()
#         mae = trainer.run_fold()
#         subject_scores[subject] = mae
#
#
#     # average across all subjects
#     mean_mae = sum(subject_scores.values()) / len(subject_scores)
#     return subject_scores, mean_mae
#
#
