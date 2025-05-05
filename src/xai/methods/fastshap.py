# xai/methods/fastshap.py

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge


def fastshap(
    model,
    dataloader,
    feature_names,
    device="cuda",
    n_samples=100,
    actif_variant="inv"
):
    """
    FastSHAP explanation using a linear surrogate model and ACTIF aggregation.

    Args:
        model: PyTorch model.
        dataloader: DataLoader yielding [B, T, F] inputs.
        feature_names: List of input feature names.
        device: Device to use.
        n_samples: Number of binary mask samples for approximation.
        actif_variant: Aggregation method ('inv', 'mean', 'meanstd', 'robust').

    Returns:
        List of feature importance dicts sorted by attribution.
    """

    model.eval()
    model.to(device)
    all_importances = []

    for inputs, _ in tqdm(dataloader, desc="FastSHAP"):
        inputs = inputs.to(device)
        batch_size, T, F = inputs.shape

        with torch.no_grad():
            base_preds = model(inputs).mean(dim=1).cpu().numpy()  # [B]

        for b in range(batch_size):
            x = inputs[b]  # [T, F]
            x_np = x.cpu().numpy()

            # Generate binary masks
            Z = np.random.binomial(1, 0.5, size=(n_samples, F))  # [n_samples, F]
            masked_inputs = np.array([x_np * z[np.newaxis, :] for z in Z])  # [n_samples, T, F]

            X_surrogate = masked_inputs.mean(axis=1)  # [n_samples, F]

            masked_tensor = torch.tensor(masked_inputs, dtype=torch.float32).to(device)
            with torch.no_grad():
                y_preds = model(masked_tensor).mean(dim=1).cpu().numpy()  # [n_samples]

            # Fit surrogate model
            surrogate = Ridge(alpha=1e-3)
            surrogate.fit(X_surrogate, y_preds)

            all_importances.append(surrogate.coef_)

    if not all_importances:
        print("[WARN] No importances computed.")
        return []

    attributions = np.array(all_importances)  # [N, F]

    # ============================
    # ACTIF aggregation strategies
    # ============================

    if actif_variant == "inv":
        importance = inverse_weighting(attributions)
    elif actif_variant == "mean":
        importance = np.mean(np.abs(attributions), axis=0)
    elif actif_variant == "meanstd":
        mean = np.mean(np.abs(attributions), axis=0)
        std = np.std(attributions, axis=0)
        importance = mean / (std + 1e-6)
    elif actif_variant == "robust":
        median = np.median(np.abs(attributions), axis=0)
        iqr = np.percentile(attributions, 75, axis=0) - np.percentile(attributions, 25, axis=0)
        importance = median / (iqr + 1e-6)
    else:
        raise ValueError(f"Unknown ACTIF variant: {actif_variant}")

    df = pd.DataFrame([importance], columns=feature_names)
    sorted_df = df.abs().T.sort_values(by=0, ascending=False).reset_index()
    sorted_df.columns = ["feature", "attribution"]

    return sorted_df.to_dict(orient="records")


def inverse_weighting(activation):
    """Reward high magnitude and low variability."""
    activation_abs = np.abs(activation)
    mean = np.mean(activation_abs, axis=0)
    std = np.std(activation_abs, axis=0)

    normalized_mean = (mean - np.min(mean)) / (np.max(mean) - np.min(mean) + 1e-6)
    inverse_std = 1 - (std - np.min(std)) / (np.max(std) - np.min(std) + 1e-6)

    return (normalized_mean + inverse_std) / 2
