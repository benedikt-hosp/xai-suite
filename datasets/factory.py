def get_dataset(name, params):
    if name == "robustvision":
        from datasets.robustvision.loader import RobustVisionDataset
        return RobustVisionDataset(**params)
    # elif name == "giw":
    #     from datasets.giw.loader import GIWDataset
    #     return GIWDataset(**params)
    # elif name == "tufts":
    #     from datasets.tufts.loader import TuftsDataset
    #     return TuftsDataset(**params)
    else:
        raise ValueError(f"Unknown dataset name: {name}")