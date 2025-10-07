#!/bin/bash

# Setup script for Stroke Rehabilitation Analysis Pipeline

echo "🚀 Setting up Stroke Rehabilitation Analysis Pipeline..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Download RTMPose models (optional)
echo "🤖 Downloading RTMPose models..."
mkdir -p models
cd models

# Download RTMPose checkpoint if not exists
if [ ! -f "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth" ]; then
    echo "Downloading RTMPose checkpoint..."
    wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.pth
fi

# Download RTMPose config if not exists
if [ ! -f "rtmpose-x_8xb256-700e_halpe26-384x288.py" ]; then
    echo "Downloading RTMPose config..."
    wget https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-x_8xb256-700e_body8-halpe26-384x288.py
fi

cd ..

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p pipeline_results/motion_analysis
mkdir -p pipeline_results/contact_detection
mkdir -p pipeline_results/generated_videos
mkdir -p pipeline_results/final_results
mkdir -p batch_results

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x main.py
chmod +x batch_process.py
chmod +x example_usage.py

echo "✅ Setup completed!"
echo ""
echo "To activate the environment:"
echo "source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "python main.py --video_path <video> --label_path <labels> --activity <activity>"
echo ""
echo "To run batch processing:"
echo "python batch_process.py --metadata_csv cleaned_metadata.csv --video_base_path <path> --label_base_path <path>"
echo ""
echo "For examples:"
echo "python example_usage.py"
