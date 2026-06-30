"""
Hybrid SIFT + VGG16 ImageNet feature concatenation experiments.

Tests three weighting configurations: equal (0.5/0.5), SIFT=0.3, SIFT=0.7.
Analysis: multi-k ARI/NMI/CER for each configuration.
"""

import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision import models
from sklearn.preprocessing import normalize
from sift import load_data as load_data_sift, extract_sift, build_bovw, VOCAB_SIZE
from vgg16 import extract_vgg_features, BASE_DIR
from utils import evaluate_multi_k
import os
import warnings
warnings.filterwarnings("ignore")


CSV_PATH = "output/ground_truth_clean.csv"


def main():
    # Load data
    images, labels, df = load_data_sift()

    # Extract SIFT features
    print("\n=== Extracting SIFT features ===")
    all_descriptors = extract_sift(images)
    sift_features = build_bovw(all_descriptors, VOCAB_SIZE)
    print(f"SIFT features: {sift_features.shape}")

    # Extract VGG16 ImageNet features
    print("\n=== Extracting VGG16 ImageNet features ===")
    model = models.vgg16(pretrained=True)
    model = nn.Sequential(*list(model.features), model.avgpool)
    imgnet_features, valid_indices = extract_vgg_features(df, BASE_DIR, model)
    print(f"VGG16 features: {imgnet_features.shape}")

    # Normalize individually before concatenation
    sift_norm = normalize(sift_features, norm="l2")
    vgg_norm = normalize(imgnet_features, norm="l2")

    # Equal weighting (0.5 / 0.5)
    hybrid_eq = normalize(np.hstack([sift_norm, vgg_norm]), norm="l2")
    evaluate_multi_k(hybrid_eq, labels, df, "Hybrid (equal)")

    # SIFT=0.3, VGG=0.7
    hybrid_03 = normalize(np.hstack([sift_norm * 0.3, vgg_norm * 0.7]), norm="l2")
    evaluate_multi_k(hybrid_03, labels, df, "Hybrid (SIFT=0.3)")

    # SIFT=0.7, VGG=0.3
    hybrid_07 = normalize(np.hstack([sift_norm * 0.7, vgg_norm * 0.3]), norm="l2")
    evaluate_multi_k(hybrid_07, labels, df, "Hybrid (SIFT=0.7)")


if __name__ == "__main__":
    main()
