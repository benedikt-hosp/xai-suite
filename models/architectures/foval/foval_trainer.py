import json
import pickle
import os

import torch
import numpy as np
from torch import nn
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from torch.cuda.amp import autocast, GradScaler

from datasets import AbstractDatasetClass
from models.architectures.foval.foval import FOVAL
from models.architectures.foval.utilities import create_optimizer



class FOVALTrainer:
    def __init__(self, dataset: AbstractDatasetClass, feature_names,
                 save_intermediates_every_epoch, device="cuda:0"):
        """
        Initialize the FOVALTrainer with feature count, dataset object, and model save path.
        """
        self.val_loader = None
        self.device =  device  # Replace 0 with the device number for your other GPU
        self.current_fold = None
        self.hidden_layer_size = None
        self.dropout_rate = None
        self.fc1_dim = None
        self.patience_counter = 0
        self.early_stopping_threshold = 1.0
        self.patience_limit = 150  # originally 150
        self.hyperparameters = None
        self.feature_names = None
        self.dataset = dataset  # Dataset object from RobustVisionDataset
        self.save_path = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.fold_results = []
        self.best_metrics = {"smae": float('inf'), "mse": float('inf'), "mae": float('inf')}
        self.current_metrics = {"val_smae": float('inf'), "val_mse": float('inf'), "val_mae": float('inf'), "train_smae": float('inf'), "train_mse": float('inf'), "train_mae": float('inf')}
        self.target_scaler = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.sequence_length = self.dataset.sequence_length
        self.csv_filename = "training_results.csv"
        # print(f"Device is: {self.device}")
        self.n_splits = len(self.dataset.subject_list)
        # print("Number of splits: ", self.n_splits)

        self.beta = None
        self.weight_decay = None
        self.learning_rate = None
        self.batch_size = None
        self.l1_lambda = None

        # Flags
        self.save_intermediates_every_epoch = save_intermediates_every_epoch

    def set_dataloaders(self, train_loader, val_loader, test_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def set_features(self, input_features):
        print("Foval trainer set features saiys: ", len(input_features))
        self.feature_count = len(input_features)  # as we need to remove target and subject column
        self.feature_names = input_features
        self.dataset.input_features = input_features

    def setup(self):
        self.initialize_model()

        self.batch_size = 460
        self.learning_rate = 0.032710957748580696
        self.weight_decay = 0.09068313284126414
        self.fc1_dim =1763
        self.dropout_rate = 0.24506232752850068
        self.hidden_layer_size = 1435
        self.beta = 0.75


    def initialize_model(self):
        self.model = FOVAL(input_size=len(self.dataset.input_features), embed_dim=1435,
                           fc1_dim=1763,
                           dropout_rate=0.24506232752850068).to(self.device)

    def run_fold(self):
        num_epochs = 200
        best_val = None  # ← this starts out None
        self.target_scaler = self.dataset.target_scaler
        self.optimizer, self.scheduler = create_optimizer(
            model=self.model, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay)

        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as profiler:
            for epoch in range(num_epochs):
                self.train_epoch(epoch)

                if self.val_loader:
                    is_last_epoch = (epoch == num_epochs - 1)
                    self.validate_epoch(epoch, is_last_epoch=is_last_epoch)
                else:
                    print("no test loader")

                self.scheduler.step()

                # if self.check_early_stopping(epoch):
                #     break
            profiler.step()  # Step profiler at the end of each epoch


        return self.best_metrics["mae"]

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        """
        if len(self.train_loader) == 0:
            print("[Warning] Train loader is empty!")
            return
        scaler = GradScaler()  # Initialize GradScaler for mixed precision

        mse_loss_fn = nn.MSELoss(reduction='sum').to(self.device)
        mae_loss_fn = nn.L1Loss().to(self.device)
        smae_loss_fn = nn.SmoothL1Loss(beta=0.75).to(self.device)

        self.model.train()
        total_samples = 0.0
        total_mae, total_mse, total_smae = 0, 0, 0

        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()

            # Example forward pass with mixed precision
            with autocast():
                y_pred = self.model(X_batch)
                # print("y-pred shape: ", y_pred.shape)
                # print("y-batch shape: ", y_batch.shape)
                smae_loss = smae_loss_fn(y_pred, y_batch)
            # Scaled backward pass
            scaler.scale(smae_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()


            # Inverse transform for metric calculation (post-backpropagation)
            y_pred_inv = self.inverse_transform_target(y_pred)
            y_batch_inv = self.inverse_transform_target(y_batch)

            # Accumulate metrics on the original scale
            total_mae += mae_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_mse += mse_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_smae += smae_loss_fn(y_pred_inv, y_batch_inv).item() * y_batch.size(0)
            total_samples += y_batch.size(0)

        if total_samples == 0:
            print("⚠️ No validation samples found. Skipping metric computation.")
            return

        self.current_metrics["train_mae"] = total_mae / total_samples
        self.current_metrics["train_mse"] = total_mse / total_samples
        self.current_metrics["train_smae"] = total_smae / total_samples
        print("Current train metrics: ", self.current_metrics)

    def validate_epoch(self, epoch, is_last_epoch=False):
        if len(self.val_loader) == 0:
            print("[Warning] Validation loader is empty!")
            return

        mse_loss_fn = nn.MSELoss(reduction='sum').to(self.device)
        mae_loss_fn = nn.L1Loss().to(self.device)
        smae_loss_fn = nn.SmoothL1Loss().to(self.device)

        self.model.eval()
        total_val_mae, total_val_mse, total_val_smae = 0, 0, 0
        total_val_samples = 0.0
        all_predictions, all_true_values = [], []

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)

                y_pred = self.inverse_transform_target(y_pred)
                y_batch = self.inverse_transform_target(y_batch)

                total_val_mae += mae_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_mse += mse_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_smae += smae_loss_fn(y_pred, y_batch).item() * y_batch.size(0)
                total_val_samples += y_batch.size(0)

                all_predictions.append(y_pred.cpu().numpy())
                all_true_values.append(y_batch.cpu().numpy())

        # Store metrics for this epoch
        self.current_metrics["val_mae"] = total_val_mae / total_val_samples
        self.current_metrics["val_mse"] = total_val_mse / total_val_samples
        self.current_metrics["val_smae"] = total_val_smae / total_val_samples
        self.current_metrics["val_r2"] = r2_score(np.concatenate(all_true_values), np.concatenate(all_predictions))
        print("Current val metrics: ", self.current_metrics)


    def inverse_transform_target(self, y_transformed):
        """
        Apply inverse transformation to the target variable.
        """
        if y_transformed.is_cuda:
            y_transformed = y_transformed.cpu()

        y_transformed_np = y_transformed.detach().numpy().reshape(-1, 1)
        if self.dataset.target_scaler:
            y_inverse_transformed = self.dataset.target_scaler.inverse_transform(y_transformed_np).flatten()
            return torch.from_numpy(y_inverse_transformed).to(self.device)
        return y_transformed


    def reset_metrics(self):
        self.best_metrics = {"smae": float('inf'), "mse": float('inf'), "mae": float('inf')}
