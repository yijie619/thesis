"""
Entropy-based unsupervised feature weighting on VGG16 ImageNet features.

First compares three weighting strategies (variance, kurtosis, entropy) at k=37/39/50.
Then runs full qualitative analysis for the best strategy (entropy):
confusion matrix, t-SNE, and multi-k ARI/NMI/CER.
"""

import numpy as np
import pandas as pd
import cv2
import torch.nn as nn
from torchvision import models
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import kurtosis
from vgg16 import load_data, extract_vgg_features, BASE_DIR
from utils import (evaluate_multi_k, plot_confusion_matrix, plot_tsne,
                   RANDOM_SEED)
import os
import warnings
warnings.filterwarnings("ignore")


CSV_PATH = "output/ground_truth_clean.csv"
SAVE_DIR = "analysis"



def compute_entropy(features, n_bins=20):
    """Compute per-dimension entropy of feature distributions."""
    entropies = []
    for j in range(features.shape[1]):
        hist, _ = np.histogram(features[:, j], bins=n_bins, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist))
        entropies.append(entropy)
    return np.array(entropies)


def apply_weighting(features, weights):
    """Apply per-dimension weights and L2-normalize."""
    weights = np.nan_to_num(weights, nan=0.0)
    max_w = weights.max()
    if max_w == 0:
        max_w = 1.0
    weights = weights / max_w
    weighted = features * weights[np.newaxis, :]
    weighted = np.nan_to_num(weighted, nan=0.0)
    return normalize(weighted, norm="l2")


def cluster_and_evaluate(features, labels, k):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    pred = kmeans.fit_predict(features)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    return pred, ari, nmi


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    images, labels, df = load_data()

    # Extract VGG16 ImageNet features
    print("\n=== Extracting VGG16 ImageNet features ===")
    model = models.vgg16(pretrained=True)
    model = nn.Sequential(*list(model.features), model.avgpool)
    imgnet_features, valid_indices = extract_vgg_features(df, BASE_DIR, model)
    imgnet_labels = df.loc[valid_indices, "label"].values
    print(f"Feature matrix: {imgnet_features.shape}")

    # Compare
    print("\n=== Weighting Strategy Comparison ===")

    # Variance-based
    weights_var = np.var(imgnet_features, axis=0)
    vgg_var = apply_weighting(imgnet_features, weights_var)
    print("\nVariance-based:")
    for k in [37, 39, 50]:
        _, ari, nmi = cluster_and_evaluate(vgg_var, imgnet_labels, k)
        print(f"  k={k}: ARI={ari:.4f}, NMI={nmi:.4f}")

    # Kurtosis-based
    weights_kurt = np.abs(kurtosis(imgnet_features, axis=0))
    vgg_kurt = apply_weighting(imgnet_features, weights_kurt)
    print("\nKurtosis-based:")
    for k in [37, 39, 50]:
        _, ari, nmi = cluster_and_evaluate(vgg_kurt, imgnet_labels, k)
        print(f"  k={k}: ARI={ari:.4f}, NMI={nmi:.4f}")

    # Entropy-based
    weights_entropy = compute_entropy(imgnet_features)
    vgg_entropy = apply_weighting(imgnet_features, weights_entropy)
    print("\nEntropy-based:")
    for k in [37, 39, 50]:
        _, ari, nmi = cluster_and_evaluate(vgg_entropy, imgnet_labels, k)
        print(f"  k={k}: ARI={ari:.4f}, NMI={nmi:.4f}")

    # Full analysis for entropy (best strategy)
    print("\n=== Entropy-based Feature Weighting: Full Analysis ===")

    evaluate_multi_k(vgg_entropy, imgnet_labels, df, "Entropy-based Weighting")

    # Qualitative analysis at k=37
    kmeans = KMeans(n_clusters=37, random_state=RANDOM_SEED, n_init=10)
    pred = kmeans.fit_predict(vgg_entropy)
    plot_confusion_matrix(pred, imgnet_labels, 37, "Attention_Entropy", SAVE_DIR)
    plot_tsne(vgg_entropy, imgnet_labels, "Attention_Entropy", SAVE_DIR)

    print(f"\nAll outputs saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
