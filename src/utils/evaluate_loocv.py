import torch
from sklearn.metrics import r2_score
from torch import GradScaler, nn, autocast
from torch.utils.data import DataLoader, Subset

from models.architectures.foval.foval_trainer import FOVALTrainer
# src/utils/evaluate_loocv.py

from torch import no_grad
from copy import deepcopy

def run_loocv(trainer_factory, dataset):
    """
    Leave-one-subject-out cross-validation loop.
    trainer_factory:  zero-arg callable that builds a fresh Trainer
    dataset:          your RobustVisionDataset instance
    """
    subject_scores = {}
    for subject in dataset.subject_list:
        # build new loaders for this fold
        train_loader, val_loader, input_dim = dataset.get_data_loader(
            train_index=[s for s in dataset.subject_list if s != subject],
            val_index=[subject],
            batch_size=dataset.batch_size  # assume you stored batch_size in dataset
        )

        # new trainer—for clean weights/scheduler each fold
        # trainer = trainer_factory(input_dim)
        # # train & validate
        # mae = trainer.run_fold(train_loader, val_loader)
        # subject_scores[subject] = mae

        trainer = trainer_factory(input_dim)
        trainer.set_features(dataset.input_features)
        trainer.set_dataloaders(train_loader, val_loader)
        trainer.setup()
        mae = trainer.run_fold()
        subject_scores[subject] = mae


    # average across all subjects
    mean_mae = sum(subject_scores.values()) / len(subject_scores)
    return subject_scores, mean_mae


