# src/utils/torchtrainer.py

import logging
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.factory import get_model

# set up module logger
logger = logging.getLogger(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TorchTrainer:
    """
    A minimal PyTorch trainer for LOOCV:
      - get fresh model per fold
      - simple MAE loss, Adam optimizer
      - train for fixed number of epochs
    """
    def __init__(
        self,
        model_name: str,
        input_size: int,
        epochs: int = 20,
        lr: float = 1e-3,
        **model_params
    ):
        logger.info(f"[Trainer Init] model={model_name}, input_size={input_size}, "
                    f"epochs={epochs}, lr={lr}, model_params={model_params}")
        # instantiate model
        self.model = get_model(
            model_name,
            input_size=input_size,
            **model_params
        ).to(device)
        self.epochs = epochs
        self.lr = lr

    def set_features(self, feature_names):
        self.feature_names = feature_names
        logger.debug(f"[Trainer] feature_names set ({len(feature_names)}): {feature_names}")

    def set_dataloaders(self, train_loader: DataLoader, val_loader: DataLoader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        logger.info(f"[Trainer] train_loader size={len(train_loader.dataset)}, "
                    f"val_loader size={len(val_loader.dataset)}")

    def setup(self):
        # define loss and optimizer
        self.criterion = nn.L1Loss()  # MAE
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        logger.debug("[Trainer] Criterion and optimizer initialized")

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in self.train_loader:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            out = self.model(x)

            # log shapes on first batch
            if n_batches == 0:
                logger.debug(f"[Epoch {epoch_idx}] x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}, out.shape={tuple(out.shape)}")

            loss = self.criterion(out.squeeze(), y)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / n_batches
        logger.info(f"[Epoch {epoch_idx}] Training MAE: {avg_loss:.4f}")

    def evaluate(self) -> float:
        self.model.eval()
        total_err = 0.0
        n_samples = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(device), y.to(device)
                out = self.model(x)
                total_err += torch.abs(out.squeeze() - y).sum().item()
                n_samples += y.size(0)

        val_mae = total_err / n_samples
        logger.info(f"[Evaluation] Validation MAE: {val_mae:.4f}")
        return val_mae

    def run_fold(self, fold_idx: int) -> float:
        logger.info(f"--- Starting fold {fold_idx} ---")
        self.setup()
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch)
            mae = self.evaluate()
        logger.info(f"*** Fold {fold_idx} complete — MAE={mae:.4f} ***")
        return mae


import logging

logger = logging.getLogger(__name__)
import logging
from src.utils.torchtrainer import TorchTrainer

logger = logging.getLogger(__name__)

def build_trainer(model_cfg):
    def factory(input_dim, feature_names=None):
        logger.info(f"[Factory] Building trainer: input_dim={input_dim}, model_cfg={model_cfg}")
        trainer = TorchTrainer(
            model_name=model_cfg["name"],
            input_size=input_dim,
            epochs=model_cfg.get("params", {}).get("epochs", 20),
            lr=model_cfg.get("params", {}).get("learning_rate", model_cfg.get("params", {}).get("lr", 1e-3)),
            **{k: v for k, v in model_cfg.get("params", {}).items() if k not in ("epochs","learning_rate","lr")}
        )
        if feature_names is not None:
            trainer.set_features(feature_names)
        return trainer
    return factory