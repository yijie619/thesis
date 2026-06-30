"""
VGG16 feature extraction (ImageNet pre-trained + Omniglot fine-tuned)
            + clustering evaluation + qualitative analysis.

ImageNet analysis: multi-k ARI/NMI/CER, confusion matrix, t-SNE, cluster samples.
Omniglot analysis: multi-k ARI/NMI/CER.
Also exports extract_vgg_features() for use by hybrid.py and attention.py.

Requires GPU for efficient feature extraction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import numpy as np
import pandas as pd
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
BATCH_SIZE = 64
OMNIGLOT_EPOCHS = 10
OMNIGLOT_N_CLASSES = 964


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_data():
    df = pd.read_csv(CSV_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/")
    images = []
    for _, row in df.iterrows():
        path = os.path.join(BASE_DIR, row["image_path"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
    labels = df["label"].values
    print(f"Loaded {len(images)} images, {len(set(labels))} classes")
    return images, labels, df


def extract_vgg_features(df, base_dir, model, batch_size=BATCH_SIZE):
    """Extract features from VGG16 convolutional layers. Returns L2-normalized features."""
    model = model.to(device)
    model.eval()
    all_features, valid_indices = [], []
    batch_imgs, batch_indices = [], []

    for idx, row in df.iterrows():
        path = os.path.join(base_dir, row["image_path"])
        try:
            img = Image.open(path).convert("L")
            img_t = transform(img)
            batch_imgs.append(img_t)
            batch_indices.append(idx)
        except:
            continue

        if len(batch_imgs) == batch_size:
            batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                feats = model(batch)
            feats = feats.view(feats.size(0), -1).cpu().numpy()
            all_features.append(feats)
            valid_indices.extend(batch_indices)
            batch_imgs, batch_indices = [], []

    if batch_imgs:
        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model(batch)
        feats = feats.view(feats.size(0), -1).cpu().numpy()
        all_features.append(feats)
        valid_indices.extend(batch_indices)

    features = normalize(np.vstack(all_features), norm="l2")
    return features, valid_indices


def train_omniglot(save_path="vgg16_omniglot.pth"):
    """Fine-tune VGG16 on Omniglot dataset starting from ImageNet weights."""
    train_dataset = datasets.Omniglot(root="omniglot", background=True,
                                       transform=transform, download=True)
    print(f"Omniglot training: {OMNIGLOT_N_CLASSES} classes, {len(train_dataset)} samples")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, OMNIGLOT_N_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(OMNIGLOT_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(lbls).sum().item()
            total += lbls.size(0)
        acc = correct / total * 100
        print(f"Epoch {epoch + 1}/{OMNIGLOT_EPOCHS}: "
              f"loss={total_loss / len(train_loader):.4f}, acc={acc:.1f}%")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Using device: {device}")

    images, labels, df = load_data()

    # VGG16 imagenet
    print("\n=== VGG16 ImageNet ===")
    model_imgnet = models.vgg16(pretrained=True)
    model_imgnet = nn.Sequential(*list(model_imgnet.features), model_imgnet.avgpool)

    print("Extracting ImageNet features...")
    imgnet_features, valid_indices = extract_vgg_features(df, BASE_DIR, model_imgnet)
    imgnet_labels = df.loc[valid_indices, "label"].values
    print(f"Feature matrix: {imgnet_features.shape}")

    evaluate_multi_k(imgnet_features, imgnet_labels, df, "VGG16 ImageNet")

    # Qualitative analysis for ImageNet
    kmeans = KMeans(n_clusters=37, random_state=RANDOM_SEED, n_init=10)
    pred = kmeans.fit_predict(imgnet_features)
    plot_confusion_matrix(pred, imgnet_labels, 37, "VGG16_ImageNet", SAVE_DIR)
    plot_tsne(imgnet_features, imgnet_labels, "VGG16_ImageNet", SAVE_DIR)
    plot_cluster_samples(pred, imgnet_labels, images, 37, "VGG16_ImageNet",
                         SAVE_DIR, mode="best_worst")

    # VGG16 Omniglot
    print("\n=== VGG16 Omniglot ===")
    model_omni = train_omniglot()

    # Remove classifier, keep convolutional features only
    model_omni_feat = nn.Sequential(*list(model_omni.features), model_omni.avgpool)

    print("Extracting Omniglot features...")
    omni_features, valid_indices_omni = extract_vgg_features(df, BASE_DIR, model_omni_feat)
    omni_labels = df.loc[valid_indices_omni, "label"].values
    print(f"Feature matrix: {omni_features.shape}")

    evaluate_multi_k(omni_features, omni_labels, df, "VGG16 Omniglot")

    print(f"\nAll outputs saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
