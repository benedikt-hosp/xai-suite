# dataset_loader_cached.py
import torch
import json
import pickle

# class PreprocessedDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir):
#         self.features = torch.load(f"{root_dir}/features.pt")
#         self.targets = torch.load(f"{root_dir}/targets.pt")
#         with open(f"{root_dir}/subject_ids.json") as f:
#             self.subject_ids = json.load(f)
#         with open(f"{root_dir}/feature_names.json") as f:
#             self.feature_names = json.load(f)
#         with open(f"{root_dir}/transformers.pkl", "rb") as f:
#             self.scalers = pickle.load(f)
#
#     def __len__(self):
#         return len(self.targets)
#
#     def __getitem__(self, idx):
#         return self.features[idx], self.targets[idx]
import torch
import json
import pickle

class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.features_tensor = torch.load(f"{root_dir}/features.pt")
        self.targets = torch.load(f"{root_dir}/targets.pt")

        with open(f"{root_dir}/subject_ids.json") as f:
            self.subject_ids = json.load(f)
        with open(f"{root_dir}/feature_names.json") as f:
            self.feature_names = json.load(f)
        with open(f"{root_dir}/transformers.pkl", "rb") as f:
            self.scalers = pickle.load(f)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features_tensor[idx], self.targets[idx]
