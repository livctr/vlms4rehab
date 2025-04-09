# cvfm4rehab

Applying modern computer vision foundation models to stroke rehabilitation tasks.

### Set Up
*Needs pytorch >=2.5.1, torchvision>=0.20.0*


Running LLaVA-OneVision, LLaVANexT-Video, InternVL

```python

conda create -n cvfm4rehab_llava python=3.10
conda activate cvfm4rehab_llava
pip install -e .[metrics,llava_next]
pip install deepspeed
cd LLaVA-NeXT ; pip install -e . ; cd ..
```

<!-- `./environment_setup.sh cvfm4rehab_vila` -> NVIDIA demo works, FloatPointQuantizeTorch is installed -->
<!-- Then, if you do `pip install -e .[metrics]`, the environment gets messed up.
Solution 1: manually check which packages are needed and add them to the VILA pyproject toml
Solution 2: first install `lmms_eval`, then "specify it" with VILA.
Try solution 2 first. -->


Running LongVILA, NVILA

```python

conda create -n cvfm4rehab_vila python=3.10
conda init
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

```

Follow [LongVA GitHub](https://github.com/EvolvingLMMs-Lab/LongVA) to create the LongVA environment.
```python

pip install -e .
pip install python-Levenshtein

# LLaVA
cd LLaVA-NeXT
pip install --upgrade pip
pip install -e ".[train]"
cd ..

```

### Reproduce the Results

To reproduce the paper experiments, manually update  `model_configs.yaml` (for logging and model settings), `mv evaluate.sh.example evaluate.sh`, and manually update the `evaluate.sh` bash file. Plots are saved in the `visualization/plots` directory.


### Important Files

- `data/utils_strokerehab.py` contains utilities for working with the labels and data paths.
- `python -m data.metadata.write_all_metadata` generates a metadata file about the video-label pairs.
- `/gpfs/data/schambralab/quantitativeRehabilitation/__data/metadata/metadata.csv` has length 5655.
    - 71 unique patients (51 stroke-impaired and 20 healthy subjects)
    - 11 activities
    - 5 repetitions
    - 2 camera angles
    - Minus a few
    - Note: a particular labeling may be `S00027_feeding1_2`, which signals a stroke patient (from
      the `S`, rather than a `C` for a control patient) with the `00027` identifier doing activity
      `feeding`. This is the first repetition (the `1` immediately after `feeding`) at camera angle `2`.
