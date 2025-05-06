# xai/methods/feature_shuffle.py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# from src.start_experiments import logger


def feature_shuffle(model, dataloader, feature_names, baseline_type='mean', device="cuda"):
    print(f"[XAI] Using {len(feature_names)} features for attribution.")
    print("feature_names", feature_names)

    model.eval()
    model.to(device)

    num_features = len(feature_names)
    num_samples = len(dataloader.dataset)

    # Calculate baseline predictions for reference (unshuffled)
    baseline_preds = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            preds = model(x).squeeze()
            baseline_preds.append(preds.detach().cpu().numpy())
    baseline_preds = np.concatenate(baseline_preds)

    importances = np.zeros((num_samples, num_features))

    for f_idx in tqdm(range(num_features), desc="Shuffle Attribution"):
        sample_idx = 0
        for x, _ in dataloader:
            x = x.to(device)
            x_shuffled = x.clone()
            batch_size = x.size(0)

            # Shuffle this feature across batch
            perm = torch.randperm(batch_size)
            x_shuffled[:, :, f_idx] = x_shuffled[perm, :, f_idx]

            with torch.no_grad():
                shuffled_preds = model(x_shuffled).squeeze().detach().cpu().numpy()

            # Difference from baseline (mean over time)
            importances[sample_idx:sample_idx + batch_size, f_idx] = np.abs(
                shuffled_preds - baseline_preds[sample_idx:sample_idx + batch_size]
            )
            sample_idx += batch_size

    mean_importance = np.mean(importances, axis=0)
    df = pd.DataFrame([mean_importance], columns=feature_names)
    sorted_df = df.T.reset_index().rename(columns={"index": "feature", 0: "attribution"})
    sorted_df = sorted_df.sort_values(by="attribution", ascending=False)
    return sorted_df.to_dict(orient="records")
