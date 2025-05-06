from models.architectures.foval.foval import FOVAL


def get_model(name, input_size=40, **kwargs):
    if name.lower() == "foval":
        if input_size is None:
            raise ValueError("Missing input_size for FOVAL model.")
        return FOVAL(input_size=input_size)
    raise NotImplementedError(f"Model {name} not supported.")

#
# def get_model(name, input_size=None, **kwargs):
#     if name == "foval":
#         return FOVAL(input_size=input_size, **kwargs)
#     return None

#
# from models.foval import FOVAL
# from models.gru import GRUModel
# from models.tcn import TCNModel
# from models.attention import AttentionModel
# from models.cnn import CNNModel
#
# def get_model(name, input_dim, **kwargs):
#     name = name.lower()
#
#     if name == "foval":
#         return FOVALModel(input_size=input_dim, **kwargs)
#     elif name == "gru":
#         return GRUModel(input_size=input_dim, **kwargs)
#     elif name == "tcn":
#         return TCNModel(input_size=input_dim, **kwargs)
#     elif name == "attention":
#         return AttentionModel(input_size=input_dim, **kwargs)
#     elif name == "cnn":
#         return CNNModel(input_size=input_dim, **kwargs)
#     else:
#         raise ValueError(f"[ERROR] Unknown model: {name}")
