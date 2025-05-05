# xai/methods/lrp.py

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from captum.attr import LRP

def lrp(model, dataloader, feature_names, device="cuda"):
    model.eval()
    model.to(device)
    all_attributions = []

    for batch_idx, (inputs, _) in enumerate(tqdm(dataloader, desc="LRP Attribution")):
        inputs = inputs.to(device)
        explainer = LRP(model)

        try:
            attr = explainer.attribute(inputs)
            attr_np = attr.detach().cpu().numpy().sum(axis=1)  # sum over time
            all_attributions.append(attr_np)
        except Exception as e:
            print(f"[ERROR] LRP failed at batch {batch_idx}: {e}")

        del inputs, attr
        torch.cuda.empty_cache()

    if not all_attributions:
        print("[WARN] No LRP attributions computed.")
        return []

    aggregated = np.concatenate(all_attributions, axis=0)
    importance = np.mean(np.abs(aggregated), axis=0)

    df = pd.DataFrame([importance], columns=feature_names)
    sorted_df = df.abs().T.sort_values(by=0, ascending=False).reset_index()
    sorted_df.columns = ["feature", "attribution"]

    return sorted_df.to_dict(orient="records")
