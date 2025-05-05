import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from captum.attr import DeepLift

def deeplift(model, dataloader, feature_names, baseline_type="zero", device="cuda"):
    """
    Compute feature importance using DeepLIFT over a full DataLoader.

    Args:
        model: PyTorch model (already on device).
        dataloader: torch.utils.data.DataLoader with shape [B, T, F].
        feature_names: list of strings with feature names.
        baseline_type: "zeroes", "random", or "mean".
        device: torch device ("cuda" or "cpu").

    Returns:
        results: list of dicts [{feature, attribution}], sorted by descending attribution.
    """
    model.to(device)
    model.eval()
    all_attributions = []

    for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="DeepLIFT Attribution")):
        inputs = inputs.to(device)

        # Baseline definition
        if baseline_type == "zero":
            baseline = torch.zeros_like(inputs)
        elif baseline_type == "random":
            baseline = torch.randn_like(inputs)
        elif baseline_type == "mean":
            baseline = torch.mean(inputs, dim=0, keepdim=True).expand_as(inputs)
        else:
            raise ValueError(f"Unsupported baseline_type: {baseline_type}")

        was_training = model.training
        model.train()  # Needed for cuDNN RNN backward

        explainer = DeepLift(model)

        try:
            attributions = explainer.attribute(inputs, baselines=baseline)
            attributions_np = attributions.detach().cpu().numpy().sum(axis=1)  # sum over timesteps
            all_attributions.append(attributions_np)
        except RuntimeError as e:
            print(f"[ERROR] Batch {batch_idx} failed: {e}")
        finally:
            if not was_training:
                model.eval()
            del inputs, baseline
            if 'attributions' in locals():
                del attributions
            torch.cuda.empty_cache()

    if not all_attributions:
        print("[WARN] No attributions computed.")
        return []

    aggregated_attributions = np.concatenate(all_attributions, axis=0)
    importance = np.mean(np.abs(aggregated_attributions), axis=0)

    df = pd.DataFrame([importance], columns=feature_names)
    sorted_df = df.abs().T.sort_values(by=0, ascending=False).reset_index()
    sorted_df.columns = ["feature", "attribution"]

    return sorted_df.to_dict(orient="records")
