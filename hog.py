"""
HOG feature extraction + clustering evaluation + qualitative analysis.

Analysis: multi-k ARI/NMI/CER, confusion matrix, t-SNE, cluster samples, cluster contents.
"""

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from utils import (evaluate_multi_k, print_cluster_contents,
                   plot_confusion_matrix, plot_tsne, plot_cluster_samples,
                   RANDOM_SEED)
import os
import warnings
warnings.filterwarnings("ignore")


CSV_PATH = "output/ground_truth_clean.csv"
BASE_DIR = "output"
IMG_SIZE = (64, 64)
SAVE_DIR = "analysis"
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
# ================================


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/")
    images, valid_rows = [], []
    for _, row in df.iterrows():
        path = os.path.join(BASE_DIR, row["image_path"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            images.append(img)
            valid_rows.append(row)
    df = pd.DataFrame(valid_rows).reset_index(drop=True)
    labels = df["label"].values
    print(f"Loaded {len(images)} images, {len(set(labels))} classes")
    return images, labels, df


def extract_hog(images):
    """Extract HOG features from all images and L2-normalize."""
    features = []
    for img in images:
        feat = hog(img, orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   block_norm="L2-Hys")
        features.append(feat)
    features = normalize(np.array(features), norm="l2")
    print(f"HOG feature dim: {features.shape[1]}")
    return features


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    images, labels, df = load_data()
    features = extract_hog(images)

    # Multi-k evaluation (k=37,39,50)
    evaluate_multi_k(features, labels, df, "HOG")

    # Detailed analysis at k=37
    kmeans = KMeans(n_clusters=37, random_state=RANDOM_SEED, n_init=10)
    pred = kmeans.fit_predict(features)

    print_cluster_contents(pred, labels, 37)
    plot_confusion_matrix(pred, labels, 37, "HOG", SAVE_DIR)
    plot_tsne(features, labels, "HOG", SAVE_DIR)
    plot_cluster_samples(pred, labels, images, 37, "HOG", SAVE_DIR, mode="all")

    print(f"\nAll outputs saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
