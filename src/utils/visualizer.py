import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Visualizer:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def plot_attribution(self, attribution, label, index=0, method_name="xai"):
        attribution = attribution.squeeze().cpu().numpy()  # shape: [T, F] or [T]
        if attribution.ndim == 1:
            attribution = attribution[:, np.newaxis]

        # === 1. Bar Plot: Top-K feature importance ===
        feature_scores = np.mean(np.abs(attribution), axis=0)
        ranking = np.argsort(-feature_scores)
        top_k = 10
        top_feats = ranking[:top_k]
        top_scores = feature_scores[top_feats]
        top_labels = [f"F{f}" for f in top_feats]

        plt.figure(figsize=(8, 4))
        plt.barh(top_labels[::-1], top_scores[::-1])
        plt.title(f"Top {top_k} Features – Sample {index} (Label: {label})")
        plt.xlabel("Importance")
        self._save_or_show(f"{method_name}_bar_sample{index}_label{label}.png")

        # === 2. Heatmap [F x T] ===
        plt.figure(figsize=(12, 5))
        sns.heatmap(np.abs(attribution).T, cmap="viridis", xticklabels=10, yticklabels=5)
        plt.title(f"Attribution Heatmap – Sample {index} (Label: {label})")
        plt.xlabel("Time Step")
        plt.ylabel("Feature")
        self._save_or_show(f"{method_name}_heatmap_sample{index}_label{label}.png")

        # === 3. Feature-wise Curves ===
        plt.figure(figsize=(10, 4))
        for i in top_feats[:3]:
            plt.plot(attribution[:, i], label=f"Feature {i}")
        plt.title(f"Top Features Over Time – Sample {index} (Label: {label})")
        plt.xlabel("Time Step")
        plt.ylabel("Attribution")
        plt.legend()
        self._save_or_show(f"{method_name}_curves_sample{index}_label{label}.png")

    def _save_or_show(self, filename):
        if self.output_dir:
            path = os.path.join(self.output_dir, filename)
            plt.tight_layout()
            plt.savefig(path)
        else:
            plt.show()
        plt.close()