# Computer Vision & ML Environment Recreation Guide

This directory contains files to recreate your computer vision and machine learning environment **without OpenMMLab packages** (mmcv, mmengine, mmdet, mmpose).

## Files Overview

- **`recreate_cv_env.sh`** - Complete bash script for automated environment setup (NO OpenMMLab)
- **`cv_ml_requirements.txt`** - Pip requirements file with exact package versions (NO OpenMMLab)
- **`recreate_openmmlab_env.sh`** - Original script with OpenMMLab packages (if needed later)
- **`README_CV_ML_Environment.md`** - This documentation file

## Environment Details

**Environment Focus:**
- Python: 3.8
- PyTorch: 2.4.1+cu118 (CUDA 11.8)
- **NO OpenMMLab packages** (mmcv, mmengine, mmdet, mmpose)
- Computer Vision: OpenCV, Matplotlib, Scikit-learn
- ML/AI: Transformers, HuggingFace, Ultralytics YOLO
- Development: Jupyter, pytest, etc.

## Quick Start

### Method 1: Automated Script (Recommended)

```bash
cd strokerehab
./recreate_cv_env.sh
```

**Features:**
- ✅ Automated conda environment creation
- ✅ PyTorch with CUDA 11.8 support
- ✅ Computer vision dependencies (OpenCV, Matplotlib, etc.)
- ✅ ML/AI packages (Transformers, YOLO, etc.)
- ✅ Development tools (Jupyter, pytest, etc.)
- ✅ Installation verification
- ✅ Jupyter kernel registration
- ❌ **NO OpenMMLab packages**

### Method 2: Manual Installation

```bash
# Create conda environment with Python 3.8
conda create -n cv_ml_env python=3.8 -y
conda activate cv_ml_env

# Install packages from requirements file
pip install -r cv_ml_requirements.txt
```

## Key Package Categories

### 🔥 Core ML/DL Frameworks
- **PyTorch 2.4.1+cu118** with CUDA 11.8 support
- **TorchVision 0.19.1+cu118**
- **TorchAudio 2.4.1+cu118**

### 👁️ Computer Vision
- **OpenCV 4.11.0.86** - Image/video processing
- **Pillow 10.2.0** - Image manipulation
- **Matplotlib 3.7.5** - Plotting and visualization
- **Seaborn 0.13.2** - Statistical data visualization
- **Scikit-learn 1.3.2** - Machine learning library

### 🤖 Transformers & HuggingFace
- **Transformers 4.46.3** - State-of-the-art NLP models
- **Tokenizers 0.20.3** - Fast tokenization
- **HuggingFace Hub 0.30.2** - Model hub access
- **Safetensors 0.5.3** - Safe tensor serialization
- **Accelerate 1.0.1** - Distributed training

### 🎯 Object Detection (YOLO)
- **Ultralytics 8.3.120** - YOLOv8/YOLOv11 implementation
- **Ultralytics-THOP 2.0.14** - Model complexity analysis

### 🏃‍♂️ Pose Estimation Tools
- **PyCocoTools 2.0.7** - COCO dataset utilities
- **XTCocoTools 1.14.3** - Extended COCO tools
- **Munkres 1.1.4** - Hungarian algorithm
- **Chumpy 0.70** - Differentiable programming

### 🛠️ Development Tools
- **Jupyter** - Interactive notebooks
- **IPython 8.12.3** - Enhanced Python shell
- **Pytest 8.3.5** - Testing framework
- **Flake8 7.1.2** - Code linting

## Usage Instructions

### Activate Environment
```bash
conda activate cv_ml_env
```

### Verify Installation
```bash
python -c "
import torch
import cv2
import transformers
import numpy as np
print('✅ PyTorch:', torch.__version__)
print('✅ OpenCV:', cv2.__version__)
print('✅ Transformers:', transformers.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
"
```

### Use with Jupyter
```bash
jupyter notebook
# Select "CV & ML (cv_ml_env)" kernel
```

### Test Your Stroke Rehabilitation LLM Code
Your `llm_caller.py` should work with this environment:
```bash
cd strokerehab/strokerehab
python llm_caller.py
```

### Python Version Check
From the terminal, you can check your Python version with:
```bash
# Current system Python
python --version

# Python 3 specifically  
python3 --version

# Inside conda environment
conda activate cv_ml_env
python --version
```

## What's Included vs Excluded

### ✅ **Included Packages**
- PyTorch ecosystem (torch, torchvision, torchaudio)
- Computer vision (OpenCV, Pillow, Matplotlib)
- Machine learning (Scikit-learn, NumPy, SciPy)
- NLP/Transformers (HuggingFace ecosystem)
- Object detection (Ultralytics YOLO)
- Development tools (Jupyter, pytest, flake8)
- Utilities (tqdm, pandas, requests)


## Alternative Pose Estimation Options

Since MMPose is excluded, you can use these alternatives:

### 1. Ultralytics YOLO Pose
```python
from ultralytics import YOLO
model = YOLO('yolo11x-pose.pt')
results = model(image)
```

### 2. MediaPipe
```bash
pip install mediapipe
```

### 3. OpenPose (if available)
```bash
# Requires separate installation
```

## Troubleshooting

### CUDA Issues
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Package Conflicts
```bash
# Remove environment and recreate
conda remove -n cv_ml_env --all
./recreate_cv_env.sh
```

### Installing Additional Packages
```bash
conda activate cv_ml_env
pip install package_name
```

## Environment Management

### Export Current Environment
```bash
conda activate cv_ml_env
pip freeze > my_current_requirements.txt
```

### Update Packages
```bash
conda activate cv_ml_env
pip install --upgrade package_name
```

### Remove Environment
```bash
conda remove -n cv_ml_env --all
```

## Script Options

### Custom Environment Name
```bash
./recreate_cv_env.sh --env-name my_custom_env
```

### Different Python Version
```bash
./recreate_cv_env.sh --python 3.9
```

### Help
```bash
./recreate_cv_env.sh --help
```
