from src.xai.methods.integrated_gradients import integrated_gradients
from src.xai.methods.deepactif import deepactif
from src.xai.methods.shap_method import shap_method
from src.xai.methods.deeplift import deeplift
from src.xai.methods.nisp import nisp
from src.xai.methods.feature_ablation import feature_ablation
from src.xai.methods.feature_shuffle import feature_shuffle
from src.xai.methods.fastshap import fastshap


def get_xai_method(name):
    if "integrated_gradients" in name:
        return integrated_gradients
    elif "deepactif" in name:
        return deepactif
    elif "shap" in name:
        return shap_method
    elif "deeplift" in name:
        return deeplift
    elif "nisp" in name:
        return nisp
    elif "feature_ablation" in name:
        return feature_ablation
    elif "feature_shuffle" in name:
        return feature_shuffle
    # elif "LRP" in name:
    #     return lrp
    elif "fastshap" in name:
        return fastshap
    else:
        raise ValueError(f"Unknown XAI method: {name}")
