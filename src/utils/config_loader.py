import json
import yaml

def load_feature_config(path):
    with open(path) as f:
        cfg = json.load(f)
    return cfg["inputs"], cfg["target"], cfg["meta"]

