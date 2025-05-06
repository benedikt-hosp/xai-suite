from models.architectures.foval.foval_trainer import FOVALTrainer

# src/trainer_factory.py

import torch.optim as optim
from models.factory import get_model

def build_trainer(cfg_model: dict):
    """
    Returns a zero-arg factory that builds a Trainer configured
    with fresh model, optimizer, scheduler.
    """
    def factory(input_dim):
        model = get_model(cfg_model["name"], input_size=input_dim, **cfg_model["params"])
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg_model["params"]["learning_rate"],
                                weight_decay=cfg_model["params"]["weight_decay"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return FOVALTrainer(model, optimizer, scheduler, device=cfg_model.get("device", "cpu"))
    return factory
