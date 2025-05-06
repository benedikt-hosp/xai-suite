import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.factory import get_model

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
        # instantiate model
        self.model = get_model(
            model_name,
            input_size=input_size,
            **model_params
        ).to(device)
        self.epochs = epochs
        self.lr = lr

    def set_features(self, feature_names):
        # optional: store for debugging or logs
        self.feature_names = feature_names

    def set_dataloaders(self, train_loader: DataLoader, val_loader: DataLoader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def setup(self):
        # define loss and optimizer
        self.criterion = nn.L1Loss()  # MAE
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )

    def train_epoch(self):
        self.model.train()
        for x, y in self.train_loader:
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out.squeeze(), y)
            loss.backward()
            self.optimizer.step()

    def evaluate(self) -> float:
        self.model.eval()
        total_err = 0.0
        n = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(device), y.to(device)
                out = self.model(x)
                total_err += torch.abs(out.squeeze() - y).sum().item()
                n += y.size(0)
        return total_err / n

    def run_fold(self) -> float:
        # run full training and return MAE on validation
        for _ in range(self.epochs):
            self.train_epoch()
        return self.evaluate()


# src/trainer_factory.py
from src.utils.torchtrainer import TorchTrainer
from models.factory import get_model
import torch.optim as optim
import torch.nn as nn

def build_trainer(model_cfg):
    """
    Returns a zero‐arg factory that builds a fresh TorchTrainer when
    you pass in the fold’s `input_dim`.
    """
    def factory(input_dim):
        model = get_model(
            model_cfg["name"],
            input_size=input_dim,
            **model_cfg.get("params", {})
        )
        optimizer = optim.Adam(
            model.parameters(),
            lr=model_cfg.get("params", {}).get("lr", 1e-3)
        )
        loss_fn = nn.L1Loss()  # MAE
        return TorchTrainer(model_name=model_cfg["name"], input_size=input_dim, lr=model_cfg.get("params", {}).get("lr", 1e-3), epochs=300)
    return factory
