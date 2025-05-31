#!/bin/bash

# Script to recreate Computer Vision and ML environment
# Based on environment analysis but excluding OpenMMLab packages

set -e  # Exit on any error

echo "🚀 Starting Computer Vision & ML Environment Recreation"
echo "======================================================="

# Configuration
ENV_NAME="cv_ml_env"
PYTHON_VERSION="3.8"
CUDA_VERSION="11.8"  # Based on torch version 2.4.1+cu118

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        print_error "Please install Anaconda or Miniconda first"
        print_error "Visit: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "Conda found: $(conda --version)"
}

# Function to create conda environment
create_conda_env() {
    print_status "Creating conda environment with Python ${PYTHON_VERSION}..."
    
    # Remove existing environment if it exists
    if conda env list | grep -q "^${ENV_NAME}"; then
        print_warning "Environment '${ENV_NAME}' already exists. Removing it..."
        conda env remove -n "${ENV_NAME}" -y
    fi
    
    # Create new environment
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    print_success "Conda environment '${ENV_NAME}' created successfully"
}

# Function to install PyTorch with CUDA support
install_pytorch() {
    print_status "Installing PyTorch with CUDA ${CUDA_VERSION} support..."
    
    conda run -n "${ENV_NAME}" pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
    
    print_success "PyTorch installed successfully"
}

# Function to install computer vision dependencies
install_cv_deps() {
    print_status "Installing computer vision dependencies..."
    
    conda run -n "${ENV_NAME}" pip install \
        opencv-python==4.11.0.86 \
        pillow==10.2.0 \
        matplotlib==3.7.5 \
        seaborn==0.13.2 \
        scipy==1.10.1 \
        numpy==1.24.1 \
        scikit-learn==1.3.2
    
    print_success "Computer vision dependencies installed"
}

# Function to install COCO and pose estimation tools (without MMPose)
install_pose_deps() {
    print_status "Installing pose estimation dependencies..."
    
    conda run -n "${ENV_NAME}" pip install \
        pycocotools==2.0.7 \
        xtcocotools==1.14.3 \
        munkres==1.1.4 \
        chumpy==0.70 \
        json-tricks==3.17.3
    
    print_success "Pose estimation dependencies installed"
}

# Function to install ML/AI packages
install_ml_deps() {
    print_status "Installing ML/AI packages..."
    
    conda run -n "${ENV_NAME}" pip install \
        transformers==4.46.3 \
        tokenizers==0.20.3 \
        huggingface-hub==0.30.2 \
        safetensors==0.5.3 \
        accelerate==1.0.1 \
        ultralytics==8.3.120 \
        ultralytics-thop==2.0.14
    
    print_success "ML/AI packages installed"
}

# Function to install development tools
install_dev_tools() {
    print_status "Installing development tools..."
    
    conda run -n "${ENV_NAME}" pip install \
        jupyter \
        ipykernel==6.29.5 \
        ipython==8.12.3 \
        pytest==8.3.5 \
        flake8==7.1.2 \
        interrogate==1.7.0 \
        yapf==0.43.0 \
        isort==4.3.21
    
    # Register kernel for Jupyter
    conda run -n "${ENV_NAME}" python -m ipykernel install --user --name "${ENV_NAME}" --display-name "CV & ML (${ENV_NAME})"
    
    print_success "Development tools installed"
}

# Function to install additional utilities
install_utilities() {
    print_status "Installing additional utilities..."
    
    conda run -n "${ENV_NAME}" pip install \
        tqdm==4.65.2 \
        pandas==2.0.3 \
        pyyaml==6.0.2 \
        requests==2.28.2 \
        psutil==7.0.0 \
        py-cpuinfo==9.0.0 \
        rich==13.4.2 \
        termcolor==2.4.0 \
        terminaltables==3.1.10 \
        tabulate==0.9.0 \
        shapely==2.0.7 \
        addict==2.4.0
    
    print_success "Additional utilities installed"
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test imports
    conda run -n "${ENV_NAME}" python -c "
import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import transformers
print('✅ PyTorch version:', torch.__version__)
print('✅ TorchVision version:', torchvision.__version__)
print('✅ NumPy version:', np.__version__)
print('✅ OpenCV version:', cv2.__version__)
print('✅ Transformers version:', transformers.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ CUDA version:', torch.version.cuda)
    print('✅ GPU count:', torch.cuda.device_count())
else:
    print('⚠️  CUDA not available - will run on CPU')
"
    
    print_success "Installation verification completed"
}

# Function to create environment info file
create_env_info() {
    print_status "Creating environment information file..."
    
    cat > "cv_ml_env_info.txt" << EOF
Computer Vision & ML Environment Information
===========================================
Created: $(date)
Environment Name: ${ENV_NAME}
Python Version: ${PYTHON_VERSION}
CUDA Version: ${CUDA_VERSION}

Key Packages:
- PyTorch: 2.4.1+cu118
- TorchVision: 0.19.1+cu118
- OpenCV: 4.11.0.86
- Transformers: 4.46.3
- Ultralytics (YOLO): 8.3.120
- Scikit-learn: 1.3.2
- Matplotlib: 3.7.5

Note: OpenMMLab packages (mmcv, mmengine, mmdet, mmpose) are NOT included

Activation Command:
conda activate ${ENV_NAME}

Deactivation Command:
conda deactivate

To recreate this environment:
bash recreate_cv_env.sh
EOF
    
    print_success "Environment info saved to cv_ml_env_info.txt"
}

# Function to show usage instructions
show_usage() {
    echo
    echo "🎉 Environment setup completed successfully!"
    echo
    echo "To activate the environment:"
    echo "  conda activate ${ENV_NAME}"
    echo
    echo "To deactivate the environment:"
    echo "  conda deactivate"
    echo
    echo "To use with Jupyter:"
    echo "  jupyter notebook"
    echo "  (Select 'CV & ML (${ENV_NAME})' kernel)"
    echo
    echo "To test the installation:"
    echo "  conda activate ${ENV_NAME}"
    echo "  python -c \"import torch, cv2, transformers; print('All imports successful!')\""
    echo
}

# Main execution
main() {
    echo "Starting environment recreation at: $(date)"
    echo
    
    # Check prerequisites
    check_conda
    
    # Create environment and install packages
    create_conda_env
    install_pytorch
    install_cv_deps
    install_pose_deps
    install_ml_deps
    install_dev_tools
    install_utilities
    
    # Verify and finalize
    verify_installation
    create_env_info
    show_usage
    
    print_success "Environment recreation completed successfully!"
}

# Handle script arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Computer Vision & ML Environment Recreation Script"
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo "  --env-name    Environment name (default: cv_ml_env)"
    echo "  --python      Python version (default: 3.8)"
    echo "  --cuda        CUDA version (default: 11.8)"
    echo
    echo "Example:"
    echo "  $0 --env-name myenv --python 3.9"
    exit 0
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main 