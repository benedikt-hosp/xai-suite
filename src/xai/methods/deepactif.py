import torch
import numpy as np
import pandas as pd
from tqdm import tqdm



def deepactif(model, dataloader, feature_names, actif_variant='inv', device='cuda'):
    print(f"[XAI] Computing feature importances using deepactif.")
    print(f"[XAI] Using {len(feature_names)} features for attribution.")
    # print("feature_names", feature_names)
    model.eval()
    model.to(device)
    all_importances = []

    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="DeepACTIF"):
            inputs = inputs.to(device)
            inputs_np = inputs.cpu().numpy()
            mean_activation = np.mean(np.abs(inputs_np), axis=1)  # [batch, features]
            all_importances.append(mean_activation)

            del inputs
            torch.cuda.empty_cache()

    all_activations = np.concatenate(all_importances, axis=0)
    importance = inverse_weighting(all_activations)

    df = pd.DataFrame([importance], columns=feature_names)
    sorted_df = df.abs().T.sort_values(by=0, ascending=False).reset_index()
    sorted_df.columns = ["feature", "attribution"]

    return sorted_df.to_dict(orient="records")


def inverse_weighting(activation):
    activation_abs = np.abs(activation)
    mean_activation = np.mean(activation_abs, axis=0)
    std_activation = np.std(activation_abs, axis=0)

    normalized_mean = (mean_activation - np.min(mean_activation)) / (np.max(mean_activation) - np.min(mean_activation) + 1e-8)
    inverse_std = 1 - (std_activation - np.min(std_activation)) / (np.max(std_activation) - np.min(std_activation) + 1e-8)

    return (normalized_mean + inverse_std) / 2
