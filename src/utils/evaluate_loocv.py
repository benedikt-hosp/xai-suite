# src/utils/evaluate_loocv.py

import logging
from torch.utils.data import Subset, DataLoader

logger = logging.getLogger(__name__)

def run_loocv(trainer_factory, dataset, batch_size=32):
    """
    trainer_factory: fn(input_dim, feature_names) -> TorchTrainer
    dataset: must have:
      - dataset.subject_list: list of unique subject IDs (strings)
      - dataset.subject_ids: array-like of length N, giving subject ID for each sample
      - dataset.input_size: int
      - dataset.feature_names: list[str]
    """
    subject_results = []
    subject_list = dataset.subject_list
    n_folds = len(subject_list)
    logger.info(f"[LOOCV] Starting {n_folds}-fold LOOCV")

    for fold_idx, subj in enumerate(subject_list, start=1):
        logger.info(f"[LOOCV] Fold {fold_idx}/{n_folds}: held-out subject={subj!r}")

        # build train/val index lists
        all_ids = dataset.subject_ids  # e.g. numpy array of length N
        train_idx = [i for i, sid in enumerate(all_ids) if sid != subj]
        val_idx   = [i for i, sid in enumerate(all_ids) if sid == subj]
        logger.debug(f"[LOOCV]  → |train|={len(train_idx)}  |val|={len(val_idx)}")

        # wrap in Subset + DataLoader
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False)

        # new trainer
        trainer = trainer_factory(len(dataset.feature_names)) # dataset.input_size, dataset.feature_names)
        trainer.set_dataloaders(train_loader, val_loader)

        # run the fold
        mae = trainer.run_fold(fold_idx)
        logger.info(f"[LOOCV]  → Fold MAE={mae:.4f}")
        subject_results.append((subj, mae))

    mean_mae = sum(m for _, m in subject_results) / n_folds
    logger.info(f"[LOOCV] Done. mean_MAE={mean_mae:.4f}")
    return subject_results, mean_mae



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
