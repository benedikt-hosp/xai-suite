# xai/methods/nisp.py

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def nisp(model, dataloader, feature_names, version="v1", device="cuda"):
    """
    Compute NISP feature importance using different hook positions.

    Args:
        model: PyTorch model (already on device and in eval mode).
        dataloader: torch.utils.data.DataLoader, yielding (x, y).
        feature_names: list of strings with feature names.
        version: 'v1' (before_lstm), 'v2' (after_lstm), 'v3' (before_output)
        device: CUDA or CPU device.

    Returns:
        List of dicts: [{'feature': ..., 'attribution': ...}, ...]
    """
    model.eval()
    model.to(device)
    all_attributions = []
    importance_scores = torch.zeros(len(feature_names), device=device)

    activations = []

    def save_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations.append(output.detach())
        return hook

    # Choose hook based on version
    if version == "time":
        hook_name = "before_lstm"
    elif version == "memory":
        hook_name = "after_lstm"
    elif version == "accuracy":
        hook_name = "before_output"
    else:
        raise ValueError(f"Unsupported NISP version: {version}")

    # Register hook
    for name, layer in model.named_modules():
        if hook_name in ["before_lstm", "after_lstm"] and isinstance(layer, torch.nn.LSTM):
            layer.register_forward_hook(save_activation(name))
            break
        if hook_name == "before_output" and isinstance(layer, torch.nn.Linear):
            layer.register_forward_hook(save_activation(name))
            break

    for x, _ in tqdm(dataloader, desc="NISP Attribution"):
        x = x.to(device)

        with torch.no_grad():
            out = model(x)
            if out.dim() == 3:
                output_importance = out.mean(dim=1)
            else:
                output_importance = out
            reduced_output = output_importance[:, :len(feature_names)]

        for act in activations:
            if act.dim() == 3:
                layer_imp = act.sum(dim=1).mean(dim=0)[:len(feature_names)]
            elif act.dim() == 2:
                layer_imp = act.mean(dim=0)[:len(feature_names)]
            else:
                continue

            importance_scores += layer_imp * reduced_output.mean(dim=0)

        activations.clear()

        all_attributions.append(importance_scores.cpu().numpy())

    if not all_attributions:
        print("[WARN] No NISP attributions computed.")
        return []

    arr = np.array(all_attributions)
    importance = np.mean(np.abs(arr), axis=0)

    df = pd.DataFrame([importance], columns=feature_names)
    sorted_df = df.abs().T.sort_values(by=0, ascending=False).reset_index()
    sorted_df.columns = ["feature", "attribution"]

    return sorted_df.to_dict(orient="records")
