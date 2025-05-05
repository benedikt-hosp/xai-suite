from models.architectures.lstm import LSTMClassifier

def get_model(name, params):
    if name == "lstm":
        return LSTMClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {name}")