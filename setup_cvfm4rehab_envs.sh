#!/usr/bin/env bash
set -e

if [ "$1" == "-y" ]; then
    yes_flag="-y"
else
    yes_flag=""
fi

# This script creates the conda environments `cvfm4rehab_llava`, `cvfm4rehab_vila`, and `cvfm4rehab_longva`.

deactivate_all_conda_envs() {
  while [[ -n "$CONDA_DEFAULT_ENV" ]]; do
    echo "Deactivating conda environment: $CONDA_DEFAULT_ENV"
    conda deactivate
  done
}

conda init

################################# cvfm4rehab BEGIN ################################
deactivate_all_conda_envs
conda create -n cvfm4rehab python=3.10 $yes_flag
conda activate cvfm4rehab
pip install -e .[metrics,tools,qwen]
pip install ultralytics
################################# cvfm4rehab END ##################################

################################# cvfm4rehab_llava BEGIN ################################
deactivate_all_conda_envs
conda create -n cvfm4rehab_llava python=3.10 $yes_flag
conda activate cvfm4rehab_llava
pip install -e .[metrics]
pip install deepspeed
cd LLaVA-NeXT ; pip install -e . ; cd ..
pip install huggingface_hub[hf_xet]
################################# cvfm4rehab_llava END ##################################


################################# cvfm4rehab_vila BEGIN ################################
deactivate_all_conda_envs
conda create -n cvfm4rehab_vila python=3.10 $yes_flag
conda activate cvfm4rehab_vila
pip install -e .[metrics]

cd VILA

#################################################
### Copied from `./VILA/environment_setup.sh` ###
#################################################
conda install -c nvidia cuda-toolkit -y

# This is required to enable PEP 660 support
pip install --upgrade pip setuptools

# Install FlashAttention2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install VILA
pip install -e ".[train,eval]"

# Quantization requires the newest triton version, and introduce dependency issue
pip install triton==3.1.0

# numpy introduce a lot dependencies issues, separate from pyproject.yaml
# pip install numpy==1.26.4

# Replace transformers and deepspeed files
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

# Downgrade protobuf to 3.20 for backward compatibility
pip install protobuf==3.20.*
#################################################
#################################################
cd ..

################################# cvfm4rehab_vila END ################################


###################################### cvfm4rehab_longva BEGIN ##########################
deactivate_all_conda_envs
conda create -n cvfm4rehab_longva python=3.10 $yes_flag
conda activate cvfm4rehab_longva
pip install --upgrade pip
pip install -e .[metrics]

cd LongVA
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e "longva/.[train]"
pip install packaging && pip install ninja && pip install flash-attn==2.5.0 --no-build-isolation --no-cache-dir
pip install -r requirements.txt
cd ..

# LLaVA
cd LLaVA-NeXT
pip install -e ".[train]"
cd ..
################################### cvfm4rehab_longva END ###############################