import torch
import numpy as np
import pandas as pd
import shap
from tqdm import tqdm

# from src.start_experiments import logger


def shap_method(model, dataloader, feature_names, version='v1', device='cuda', sequence_length=10):
    torch.backends.cudnn.enabled = False
    print(f"[XAI] Using {len(feature_names)} features for attribution.")
    print("feature_names", feature_names)
    # Store model training mode and set to required state
    was_training = model.training
    model.train()  # Required for SHAP + cuDNN RNNs
    model.to(device)

    # SHAP variant configuration
    if version == 'v1':  # Memory-efficient
        background_size, nsamples, explainer_type = 10, 50, 'gradient'
    elif version == 'v2':  # Time-efficient
        background_size, nsamples, explainer_type = 5, 20, 'gradient'
    elif version == 'v3':  # High-precision
        background_size, nsamples, explainer_type = 50, 1000, 'deep'
    else:
        raise ValueError(f"Unknown SHAP version: {version}")

    shap_values_accumulated = []

    for input_batch, _ in tqdm(dataloader, desc=f"SHAP ({version})"):
        input_batch = input_batch.to(device)
        input_batch.requires_grad = True

        background_data = input_batch[:background_size]

        try:
            if explainer_type == 'deep':
                # explainer = shap.DeepExplainer(model, background_data)
                explainer = shap.GradientExplainer(model, background_data)

                shap_values = explainer.shap_values(input_batch)
            elif explainer_type == 'gradient':
                explainer = shap.GradientExplainer(model, background_data)
                shap_values = explainer.shap_values(input_batch)
            elif explainer_type == 'kernel':
                model_wrapper = PyTorchModelWrapper_SHAP(model)
                background_data_np = background_data.cpu().numpy()
                explainer = shap.KernelExplainer(model_wrapper, background_data_np)
                shap_values = explainer.shap_values(input_batch.cpu().numpy(), nsamples=nsamples)
            else:
                raise ValueError(f"Unsupported explainer type: {explainer_type}")

            shap_values_accumulated.append(shap_values)

        except Exception as e:
            print(f"[ERROR] SHAP failed at batch with error: {e}")
            continue

    if not shap_values_accumulated:
        print("[WARN] No SHAP values were collected.")
        return []

    # Combine SHAP values from all batches
    shap_values_np = np.concatenate(shap_values_accumulated, axis=0)
    shap_values_reshaped = shap_values_np.reshape(-1, sequence_length, len(feature_names))

    # Mean over time steps
    mean_shap_values = np.mean(np.abs(shap_values_reshaped), axis=1)

    # Inverse weighting or simple mean over all samples

    weights = 1 / (np.arange(mean_shap_values.shape[0]) + 1)
    importance = (mean_shap_values.T @ weights) / weights.sum()

    df = pd.DataFrame([importance], columns=feature_names)
    sorted_df = df.abs().T.sort_values(by=0, ascending=False).reset_index()
    sorted_df.columns = ["feature", "attribution"]

    if not was_training:
        model.eval()

    return sorted_df.to_dict(orient="records")


class PyTorchModelWrapper_SHAP:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(next(self.model.parameters()).device)
        with torch.no_grad():
            model_output = self.model(x_tensor)
        if model_output.dim() == 3:
            return model_output.mean(dim=1).cpu().numpy()
        return model_output.cpu().numpy()


class PyTorchModelWrapper_SHAP:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(next(self.model.parameters()).device)
        with torch.no_grad():
            model_output = self.model(x_tensor)
        if model_output.dim() == 3:
            return model_output.mean(dim=1).cpu().numpy()
        return model_output.cpu().numpy()
