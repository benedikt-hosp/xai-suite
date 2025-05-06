import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def create_optimizer(learning_rate, weight_decay, model=None):
    # Set a seed before initializing your model
    # seed_everything(seed=42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=learning_rate)  # 100

    return optimizer, scheduler