# xai/methods/integrated_gradients.py

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from captum.attr import IntegratedGradients
from torch.cuda.amp import autocast
import logging
logger = logging.getLogger(__name__)

def integrated_gradients(
    model,
    dataloader,
    feature_names,
    baseline_type="zeroes",
    n_steps=50,
    device="cuda"
):
    """
    Compute feature importance using Integrated Gradients over a full DataLoader.

    Args:
        model: PyTorch model (already on device and in eval mode).
        dataloader: torch.utils.data.DataLoader, yielding (x, y).
        feature_names: list of strings with feature names.
        baseline_type: "zeroes", "random", or "mean".
        steps: number of integration steps.
        actif_variant: strategy to aggregate (currently only 'mean').
        device: device to run on.

    Returns:
        List of dicts: [{'feature': ..., 'attribution': ...}, ...]
    """
    model.eval()
    # model.to(device)
    all_attributions = []

    for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="Attribution")):
        inputs = inputs.to(device)

        # Choose baseline
        if baseline_type == "zeroes":
            baseline = torch.zeros_like(inputs)
        elif baseline_type == "min":
            baseline = torch.randn_like(inputs)
            # baseline = lambda x: torch.min(x, dim=0, keepdim=True).values  # Minimum über Batch
        elif baseline_type == "mean":
            baseline = torch.mean(inputs, dim=0, keepdim=True).expand_as(inputs)
        else:
            raise ValueError(f"Unsupported baseline_type: {baseline_type}")

        # Save current mode
        was_training = model.training
        model.train()  # required for cuDNN backward to work
        ig = IntegratedGradients(model)

        try:
            with autocast():
                attributions = ig.attribute(inputs, baselines=baseline, n_steps=n_steps)
            attributions_np = attributions.detach().cpu().numpy().sum(axis=1)  # sum over time
            all_attributions.append(attributions_np)
        except RuntimeError as e:
            logger.error(f"[WARN] Failed at batch {batch_idx}: {e}")
        finally:
            torch.cuda.empty_cache()

            # Restore mode
            if not was_training:
                model.eval()

            del inputs, baseline
            if 'attributions' in locals():
                del attributions

    if not all_attributions:
        logger.warning("[WARN] No attributions computed.")

        return []

    aggregated_attributions = np.concatenate(all_attributions, axis=0)
    importance = np.mean(np.abs(aggregated_attributions), axis=0)

    df = pd.DataFrame([importance], columns=feature_names)
    sorted_df = df.abs().T.sort_values(by=0, ascending=False).reset_index()
    sorted_df.columns = ["feature", "attribution"]

    return sorted_df.to_dict(orient="records")
