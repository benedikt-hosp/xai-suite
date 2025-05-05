# utils/model_loader.py
def get_model(name, params):
    if name == "foval":
        from models.architectures.lstm import FOVAL
        # Only keep model-specific args
        model_args = {
            "dropout_rate": params["dropout_rate"],
            "embed_dim": params["embed_dim"],
            "fc1_dim": params["fc1_dim"]
        }

        return FOVAL(**model_args)
    raise ValueError(f"Unknown model: {name}")