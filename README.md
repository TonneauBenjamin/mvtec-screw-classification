# MVTec Screw Dataset - Binary Classification Project

## Project Overview

This project implements a **binary classification model** for the MVTec "screw" dataset using PyTorch with transfer learning. The model classifies screw images as either **Good** (OK) or **Defective** (NG).

### Key Features
- **Architecture**: ResNet18 with transfer learning (pre-trained on ImageNet)
- **Validation Strategy**: 5-Fold Cross-Validation
- **Class Imbalance Handling**: Focal Loss and Class Weights
- **Data Augmentation**: Rotation, Flip, Zoom transformations
- **Expected Performance**: 
  - Accuracy: ~92%
  - Recall: ~85%
  - ROC-AUC: >0.90

---

## Project Structure

```
source-code-ai-BENJAMIN-20260418/
├── requirements.txt          # Python dependencies
├── download_data.py          # Dataset download script
├── main_analysis.ipynb       # Main Jupyter Notebook with complete workflow
└── README.md                 # This file
```

---

## Quick Start Guide

### Prerequisites
- Python 3.8+
- pip or conda package manager
- ~2GB of free disk space (for dataset)
- GPU recommended (CUDA 11.8+) for faster training

### Step 1: Clone/Copy the Project
```bash
cd /path/to/source-code-ai-BENJAMIN-20260418
```

### Step 2: Install Dependencies
```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda create -n screw-classifier python=3.9
conda activate screw-classifier
pip install -r requirements.txt
```

### Step 3: Download the Dataset
```bash
python download_data.py --output-dir ./data --verify
```

This script will:
- Download the MVTec screw dataset from Google Drive (~150MB compressed)
- Extract it to `./data/mvtec_screw/`
- Optionally verify the dataset structure

**Expected output structure:**
```
data/mvtec_screw/
├── train/
│   ├── good/      (250 images)
│   └── defective/ (50 images)
└── test/
    ├── good/      (test images)
    └── defective/ (test images)
```

### Step 4: Run the Analysis Notebook
```bash
jupyter notebook main_analysis.ipynb
```

Or use VS Code's Jupyter extension to open the notebook directly.

---

## Notebook Sections

The `main_analysis.ipynb` notebook contains 5 main sections:

### 1. Setup
- Import required libraries
- Set random seeds for reproducibility
- Configure device (GPU/CPU)
- Define constants and paths

### 2. Exploratory Data Analysis (EDA)
- Dataset statistics (class distribution)
- Pixel intensity analysis
- Sample image visualization
- Class imbalance visualization (5:1 ratio)

### 3. Model Definition
- ResNet18 backbone loading (ImageNet pre-trained)
- Custom classification head (binary classifier)
- Model architecture summary
- Parameter counting

### 4. Training Loop
- 5-Fold Cross-Validation implementation
- Data augmentation setup (rotation, flip, zoom)
- Batch processing with PyTorch DataLoader
- Focal Loss + Class Weights for class imbalance
- Training/validation metrics tracking
- Per-fold results aggregation

### 5. Evaluation
- Confusion Matrix visualization
- ROC-AUC curve plotting
- Precision-Recall curve
- Classification metrics (Accuracy, Precision, Recall, F1-Score)
- Per-fold performance summary

---

## Technical Details

### Model Configuration
```python
- Base Model: ResNet18 (ImageNet pre-trained)
- Input Size: 224 × 224 pixels
- Output: Binary classification (Good / Defective)
- Optimizer: Adam (lr=1e-4)
- Loss Function: Focal Loss + Class Weights
- Batch Size: 16
- Epochs per fold: 50
```

### Class Imbalance Handling
The dataset has a **5:1 ratio** (250 good : 50 defective):
- **Approach 1**: Class Weights - Weighted loss function
- **Approach 2**: Focal Loss - Focuses on hard negatives
- **Data Augmentation**: Over-sampling minority class during training

### Cross-Validation Strategy
- **5-Fold Stratified Cross-Validation** ensures balanced class distribution in each fold
- Training set: 320 samples (80%)
- Validation set: 80 samples (20%)
- Final metrics: Average across all 5 folds

---

## Expected Results

After running the complete notebook, you should see:

| Metric | Target | Typical Range |
|--------|--------|---------------|
| Accuracy | ~92% | 88-94% |
| Recall (Defective) | ~85% | 80-90% |
| Precision | ~88% | 85-91% |
| F1-Score | ~86% | 82-89% |
| ROC-AUC | >0.90 | 0.88-0.95 |

---

## Advanced Options

### Modify Training Parameters
Edit the constants section in the notebook:
```python
SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_FOLDS = 5
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4
```

### Use Different Architecture
Change the model definition section:
```python
# Options: ResNet18, ResNet50, EfficientNetB0, MobileNetV2
model = timm.create_model('resnet18', pretrained=True, num_classes=2)
```

### Adjust Augmentation Pipeline
Modify the augmentation transforms:
```python
transforms.RandomRotation(degrees=20)
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.5)
transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))
```

---

## Data Management

### Dataset Size
- Training set: 300 images (~30MB)
- Test set: 100 images (~10MB)
- Total: ~150MB compressed, ~300MB extracted

### Storage Optimization
The project is designed to be **< 50MB** (without extracted data):
- requirements.txt: ~1KB
- download_data.py: ~5KB
- main_analysis.ipynb: ~500KB
- README.md: ~10KB

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Ensure requirements.txt is installed
```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in the notebook
```python
BATCH_SIZE = 8  # Reduced from 16
```

### Issue: "Download fails"
**Solution**: Manual download from Google Drive
```bash
# Download the file manually, then:
python download_data.py --verify
```

### Issue: "Dataset extraction fails"
**Solution**: Check free disk space and permissions
```bash
df -h  # Check disk space (need ~500MB min)
chmod -R 755 ./data  # Fix permissions if needed
```

---

## References

- **MVTec Industrial Vision Dataset**: https://www.mvtec.com/company/research/datasets
- **PyTorch Documentation**: https://pytorch.org/docs/
- **ResNet Paper**: He et al., "Deep Residual Learning for Image Recognition" (2015)
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- **Cross-Validation**: Scikit-learn documentation

---

## Checklist for Submission

- [ ] Dataset downloaded successfully
- [ ] All dependencies installed
- [ ] Notebook runs without errors
- [ ] Results match expected performance (~92% accuracy)
- [ ] All visualizations generated correctly
- [ ] Folder structure matches requirements
- [ ] Total folder size < 50MB

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure dataset is properly extracted
4. Check GPU availability and CUDA compatibility

---

**Last Updated**: April 2026  
**Project**: MVTec Screw Classification - AI Engineering Assignment  
**Student**: BENJAMIN  
**Submission Date**: 2026-04-18
