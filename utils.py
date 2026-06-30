"""
Shared utility functions for clustering evaluation and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
import os
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42


# For CER
def levenshtein(s1, s2):
    """Compute Levenshtein distance between two sequences."""
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def compute_cer(pred, labels, df, k):
    """Compute CER using majority vote mapping. Returns overall CER."""
    cluster_to_label = {}
    for c in range(k):
        mask = pred == c
        if mask.sum() == 0:
            cluster_to_label[c] = "?"
            continue
        counts = pd.Series(labels[mask]).value_counts()
        cluster_to_label[c] = counts.index[0]
    pred_labels = [cluster_to_label.get(p, "?") for p in pred]

    total_edit, total_len = 0, 0
    for (page, line_num), group in df.groupby(["page", "line"]):
        group_sorted = group.sort_values("position")
        gt_seq = group_sorted["label"].tolist()
        pred_seq = [pred_labels[i] for i in group_sorted.index]
        total_edit += levenshtein(gt_seq, pred_seq)
        total_len += len(gt_seq)
    return total_edit / total_len


# K evaluation at k= 37, 39, 50
def evaluate_multi_k(features, labels, df, method_name, ks=(37, 39, 50)):
    """Run k-means at multiple k values, print ARI, NMI, CER."""
    print(f"\n{method_name}:")
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        pred = kmeans.fit_predict(features)
        ari = adjusted_rand_score(labels, pred)
        nmi = normalized_mutual_info_score(labels, pred)
        cer = compute_cer(pred, labels, df, k)
        print(f"  k={k}: ARI={ari:.4f}, NMI={nmi:.4f}, CER={cer:.4f} ({cer * 100:.1f}%)")


# Cluser content
def print_cluster_contents(pred, labels, k):
    """Print top-3 labels per cluster with purity."""
    print("\n=== Cluster Contents ===")
    for c in range(k):
        mask = pred == c
        cluster_labels = labels[mask]
        counts = pd.Series(cluster_labels).value_counts()
        top = counts.head(3).to_dict()
        purity = counts.iloc[0] / len(cluster_labels) * 100
        print(f"Cluster {c:2d} ({mask.sum():4d} imgs, purity={purity:.0f}%): {top}")


# Confusion Matrix
def plot_confusion_matrix(pred, labels, k, method_name, save_dir):
    """Plot and save confusion matrix using Hungarian mapping."""
    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    cost_matrix = np.zeros((k, n_labels))
    for c in range(k):
        mask = pred == c
        for l in labels[mask]:
            cost_matrix[c, label_to_idx[l]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    cluster_to_label = {r: unique_labels[ci] for r, ci in zip(row_ind, col_ind)}
    mapped_pred = np.array([cluster_to_label.get(p, "?") for p in pred])

    conf_matrix = np.zeros((n_labels, n_labels), dtype=int)
    for gt, pr in zip(labels, mapped_pred):
        if pr in label_to_idx:
            conf_matrix[label_to_idx[gt], label_to_idx[pr]] += 1

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(conf_matrix, cmap="Blues")
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels(unique_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(unique_labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"{method_name} k={k} Confusion Matrix")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"confusion_{method_name}.png"), dpi=150)
    plt.close()
    return cluster_to_label


# T-SNE
def plot_tsne(features, labels, method_name, save_dir):
    """Plot and save t-SNE visualization colored by top 10 GT labels."""
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    features_2d = tsne.fit_transform(features)

    top10 = pd.Series(labels).value_counts().head(10).index.tolist()
    fig, ax = plt.subplots(figsize=(12, 10))
    for label in top10:
        mask = labels == label
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], label=label, s=5, alpha=0.6)
    other_mask = ~np.isin(labels, top10)
    ax.scatter(features_2d[other_mask, 0], features_2d[other_mask, 1],
               c="lightgray", s=3, alpha=0.3, label="other")
    ax.legend(markerscale=5, fontsize=8)
    ax.set_title(f"{method_name} t-SNE (top 10 labels)")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"tsne_{method_name}.png"), dpi=150)
    plt.close()


# Cluster samples
def plot_cluster_samples(pred, labels, images, k, method_name, save_dir,
                         mode="all"):
    """
    Plot cluster samples.
    mode="all": show all k clusters.
    mode="best_worst": show best 3 + worst 3 by purity.
    """
    # Majority vote mapping for labels
    cluster_to_label = {}
    for c in range(k):
        mask = pred == c
        counts = pd.Series(labels[mask]).value_counts()
        cluster_to_label[c] = counts.index[0]

    purities = []
    for c in range(k):
        mask = pred == c
        counts = pd.Series(labels[mask]).value_counts()
        purities.append((c, counts.iloc[0] / mask.sum()))
    purities.sort(key=lambda x: x[1])

    if mode == "best_worst":
        show_clusters = [p[0] for p in purities[:3]] + [p[0] for p in purities[-3:]]
        n_rows = 6
        fig_height = 12
        suffix = "_best_worst"
    else:
        show_clusters = list(range(k))
        n_rows = k
        fig_height = k * 2
        suffix = "_all"

    fig, axes = plt.subplots(n_rows, 8, figsize=(16, fig_height))
    for row, c in enumerate(show_clusters):
        mask = pred == c
        indices = np.where(mask)[0]
        np.random.seed(RANDOM_SEED)
        sample = np.random.choice(indices, min(8, len(indices)), replace=False)
        purity_val = [p[1] for p in purities if p[0] == c][0]
        for j in range(8):
            if j < len(sample):
                axes[row, j].imshow(images[sample[j]], cmap="gray")
                if j == 0:
                    axes[row, j].set_ylabel(
                        f"C{c}->{cluster_to_label.get(c, '?')}\n{purity_val:.0%}",
                        fontsize=7)
            axes[row, j].axis("off")
    plt.suptitle(f"{method_name} k={k}: Cluster Samples", fontsize=14)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"samples_{method_name}{suffix}.png"), dpi=100)
    plt.close()
