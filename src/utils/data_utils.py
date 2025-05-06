import random
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_optimizer(learning_rate, weight_decay, model=None):
    # Set a seed before initializing your model
    seed_everything(seed=42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=learning_rate)  # 100

    return optimizer, scheduler


def create_lstm_tensors_dataset(X, y, isTrain):
    # Check for None or empty inputs
    assert X is not None and len(X) > 0, "X is empty or None"
    assert y is not None and len(y) > 0, "y is empty or None"

    # Convert training features
    features = np.array([sequence.values for sequence in X])
    # print(f"features shape: {features.shape}")
    features_tensor = torch.tensor(features, dtype=torch.float32)

    # Convert training targets
    targets_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    if isTrain:
        print(f"Training dataset has {features_tensor.shape[0]} samples with sequence shape {features_tensor.shape[1:]} and {len(y)} labels")
    else:
        print(
            f"Validation dataset has {features_tensor.shape[0]} samples with sequence shape {features_tensor.shape[1:]} and {len(y)} labels")

    return features_tensor, targets_tensor


def create_dataloaders_dataset(features_tensor, targets_tensor, batch_size):
    train_dataset = TensorDataset(features_tensor, targets_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=False, pin_memory=True, num_workers=12, persistent_workers=True)
    return train_loader


def print_results(iteration, batch_size, embed_dim, dropoutRate, l1_lambda, learning_rate, weight_decay,
                  fc1_dim,
                  avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae, avg_train_r2,
                  best_train_mse, best_train_mae, best_train_smae, avg_val_mse, avg_val_rmse, avg_val_mae,
                  avg_val_smae, avg_val_r2, best_val_mse, best_val_mae, best_val_smae):
    """ 4. Print current run average values"""
    print(f"Run: {iteration}:\n"
          f" TRAINING: MSE:", avg_train_mse, " \tRMSE:", avg_train_rmse, " \tMAE:", avg_train_mae, " \tHuber: ",
          avg_train_smae, " \tR-squared:", avg_train_r2,
          " \tBest train MSE:", best_train_mse, "\tBest train MAE:", best_train_mae, "\t Best train Huber:",
          best_train_smae,
          " \nVALIDATION: \tMSE:", avg_val_mse, " \tRMSE:", avg_val_rmse, " \tMAE:", avg_val_mae, " \tHuber: ",
          avg_val_smae, " \tR-squared:", avg_val_r2,
          " \tBest val MSE:", best_val_mse, " \tBest Val MAE:", best_val_mae, " \tBest val Huber:", best_val_smae,
          "\n",
          " Batch size: ", batch_size, " \tDropout: ", dropoutRate, " \tembed_dim :", embed_dim, " \tfc1_dim :",
          fc1_dim,
          " \tLearning rate: ", learning_rate, "\tWeight decay: ", weight_decay, "\tL1: ", l1_lambda)
