# Exploring Feature Extraction Methods for Unsupervised Symbol Recognition in Historical Ciphers

Code for the master's thesis comparing feature extraction methods for unsupervised k-means clustering of historical cipher symbols.

## Data

The BNF cipher dataset used in this study is not publicly available. It can be obtained from the [DESCRYPT Project](https://descrypt.org/). After obtaining the data, place page images in `img/` and JSON annotation files in `BNF3029-92/`.

## Pipeline

1. **Segmentation** (`segmentation.py`): Extracts individual symbol images from page scans using page-level connected component analysis. Outputs `output/ground_truth.csv` with auto-matched and review-flagged symbols.

2. **Manual review**: The auto-matched lines were manually inspected and lines with incorrect segmentations were removed, producing `output/ground_truth_clean.csv` (300 lines, 8817 symbols). This step is not automated.

3. **Feature extraction and evaluation**: Each script below reads `ground_truth_clean.csv`, extracts features, runs k-means clustering, and outputs evaluation metrics and visualizations.

## Files

| File | Description |
|------|-------------|
| `segmentation.py` | Symbol segmentation from manuscript page images |
| `utils.py` | Shared evaluation and visualization functions (CER, confusion matrix, t-SNE, cluster samples) |
| `hog.py` | HOG feature extraction and evaluation |
| `sift.py` | SIFT + Bag of Visual Words feature extraction and evaluation |
| `vgg16.py` | VGG16 feature extraction (ImageNet pre-trained and Omniglot fine-tuned) |
| `hybrid.py` | Hybrid SIFT + VGG16 feature concatenation experiments |
| `attention.py` | Entropy-based unsupervised feature weighting on VGG16 features |

## Requirements

- Python 3.8+
- OpenCV (`opencv-contrib-python`)
- scikit-image, scikit-learn, NumPy, Pandas, Matplotlib, SciPy
- PyTorch + torchvision (for VGG16, GPU recommended)
