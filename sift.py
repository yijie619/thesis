"""
SIFT + Bag of Visual Words feature extraction + clustering evaluation
           + qualitative analysis.

Analysis: multi-k ARI/NMI/CER, confusion matrix, t-SNE, cluster samples, cluster contents.
Also exports extract_sift() and build_bovw() for use by hybrid.py.
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
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
VOCAB_SIZE = 100
SAVE_DIR = "analysis"


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


def extract_sift(images):
    """Extract SIFT descriptors from all images."""
    sift = cv2.SIFT_create()
    all_descriptors = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            all_descriptors.append(des)
        else:
            all_descriptors.append(np.zeros((1, 128), dtype=np.float32))
    print(f"SIFT: avg {np.mean([len(d) for d in all_descriptors]):.1f} keypoints per image")
    return all_descriptors


def build_bovw(all_descriptors, vocab_size):
    """Build visual vocabulary and compute BoVW histograms. Returns L2-normalized features."""
    stacked = np.vstack(all_descriptors)
    print(f"Building vocabulary: {stacked.shape[0]} descriptors -> {vocab_size} words")
    vocab = MiniBatchKMeans(n_clusters=vocab_size, random_state=RANDOM_SEED,
                            batch_size=5000, n_init=3)
    vocab.fit(stacked)
    histograms = []
    for des in all_descriptors:
        words = vocab.predict(des)
        hist, _ = np.histogram(words, bins=range(vocab_size + 1))
        histograms.append(hist.astype(np.float32))
    histograms = normalize(np.array(histograms), norm="l2")
    return histograms


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    images, labels, df = load_data()

    all_descriptors = extract_sift(images)
    features = build_bovw(all_descriptors, VOCAB_SIZE)
    print(f"Feature matrix: {features.shape}")

    # Multi-k evaluation
    evaluate_multi_k(features, labels, df, f"SIFT BoVW (vocab={VOCAB_SIZE})")

    # Detailed analysis at k=37
    kmeans = KMeans(n_clusters=37, random_state=RANDOM_SEED, n_init=10)
    pred = kmeans.fit_predict(features)

    print_cluster_contents(pred, labels, 37)
    plot_confusion_matrix(pred, labels, 37, "SIFT", SAVE_DIR)
    plot_tsne(features, labels, "SIFT", SAVE_DIR)
    plot_cluster_samples(pred, labels, images, 37, "SIFT", SAVE_DIR, mode="all")

    print(f"\nAll outputs saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
