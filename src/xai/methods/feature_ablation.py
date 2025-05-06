# xai/methods/ablation.py

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from captum.attr import FeatureAblation

# from src.start_experiments import logger


def feature_ablation(model, dataloader, feature_names, device="cuda"):

    """
    Feature ablation using Captum with ACTIF-style aggregation (mean absolute values).

    Args:
        model: PyTorch model, already moved to the correct device.
        dataloader: DataLoader providing input batches of shape [B, T, F].
        feature_names: List of feature names.
        device: Torch device to use.

    Returns:
        List of dicts: [{feature, attribution}], sorted by descending attribution.
    """

    print(f"[XAI] Using {len(feature_names)} features for attribution.")
    print("feature_names", feature_names)
    model.eval()
    model.to(device)

    all_attributions = []

    for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Ablation")):
        inputs = inputs.to(device)

        # Captum's FeatureAblation wrapper
        ablator = FeatureAblation(model)

        try:
            attr = ablator.attribute(inputs)
            attr_np = attr.detach().cpu().numpy().sum(axis=1)  # sum over timesteps
            all_attributions.append(attr_np)
            del inputs, attr

        except Exception as e:
            print(f"[ERROR] Ablation failed at batch {batch_idx}: {e}")

        torch.cuda.empty_cache()

    if not all_attributions:
        print("[WARN] No attributions computed.")
        return []

    # Aggregate
    aggregated = np.concatenate(all_attributions, axis=0)
    importance = np.mean(np.abs(aggregated), axis=0)

    df = pd.DataFrame([importance], columns=feature_names)
    sorted_df = df.abs().T.sort_values(by=0, ascending=False).reset_index()
    sorted_df.columns = ["feature", "attribution"]

    return sorted_df.to_dict(orient="records")
