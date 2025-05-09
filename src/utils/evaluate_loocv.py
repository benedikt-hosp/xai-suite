# src/utils/evaluate_loocv.py

import os
import logging

from datasets.factory import get_dataset

logger = logging.getLogger(__name__)

import numpy as np
def run_loocv(trainer_factory, dataset):
    subject_maes = {}
    batch_size, n_epochs = 460, 500
    # grab your pre‐loaded raw features & subjects
    features_raw = dataset.features_raw
    targets_raw  = dataset.targets_raw
    subject_ids  = dataset.subject_ids


    for held_out_subj  in dataset.subject_list:

        tr_loader, va_loader = dataset.get_fold_from_tensors(held_out_subj, batch_size=460)
        trainer = trainer_factory(
            input_dim=len(dataset.feature_names)
        )
        trainer.set_dataloaders(tr_loader, va_loader)
        best_mae = trainer.fit(epochs=500)
        subject_maes[held_out_subj] = best_mae


        # train_loader, val_loader = dataset.get_fold_from_df(
        #     held_out_subj,
        #     batch_size=batch_size
        # )
        #
        # trainer = trainer_factory(
        #     input_dim=len(dataset.feature_names),
        #     feature_names=dataset.feature_names
        # )
        # trainer.set_dataloaders(train_loader, val_loader)
        #
        # best_mae = trainer.fit(epochs=500)
        # subject_maes[held_out_subj] = best_mae
        # subject_maes[held_out_subj] = best_mae
        # logger.info(f"[LOOCV] Subj={held_out_subj} MAE={best_mae:.2f} cm")
