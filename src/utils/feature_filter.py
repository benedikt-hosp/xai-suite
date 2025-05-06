def get_filtered_feature_indices(feature_names, ranking, topk_percent, strategy="keep"):
    """
    Returns the indices of features to keep based on a ranking and strategy.

    Args:
        feature_names (list): All original feature names.
        ranking (list): Sorted list of dicts with 'feature' keys.
        topk_percent (int): Percentage of top-ranked features to use/remove.
        strategy (str): Either 'keep' or 'remove'.

    Returns:
        list of indices to keep in the feature tensor.
    """
    num_total = len(feature_names)
    k = max(1, int(num_total * (topk_percent / 100)))
    top_k_features = [entry["feature"] for entry in ranking[:k]]

    if strategy == "keep":
        used_features = top_k_features
    elif strategy == "remove":
        used_features = [f for f in feature_names if f not in top_k_features]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    indices = [i for i, f in enumerate(feature_names) if f in used_features]
    return indices
