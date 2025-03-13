# cvfm4rehab

Applying modern computer vision foundation models to stroke rehabilitation tasks.

### Set Up
*Needs pytorch >=2.5.1, torchvision>=0.20.0*

```python

conda create -n cvfm4rehab python=3.10
conda activate cvfm4rehab

pip install -e .
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales  # for VILA


pip install deepspeed
pip install flash-attn --no-build-isolation

# LLaVA
cd LLaVA-NeXT ; pip install -e . ; cd ..

# Metric
pip install python-Levenshtein
```


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
