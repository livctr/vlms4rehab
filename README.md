# cvfm4rehab

Applying modern computer vision foundation models to stroke rehabilitation tasks.

### Set Up

**Installation**

```bash

git clone https://github.com/livctr/cvfm4rehab.git
cd cvfm4rehab
bash ./setup_cvfm4rehab_envs.sh
```

The bash script creates three separate environments `cvfm4rehab_llava`, `cvfm4rehab_vila`, and `cvfm4rehab_longva` for running different models.

| Environment | Models |
|-------------|--------|
| `cvfm4rehab_llava` | internvl2_5_[2,8,38,78]b, llava_next_video_[7,72]b, llava_ov_[0p5,7,72]b, |
| `cvfm4rehab_vila` | nvila_[8,15]b, longvila_8b |
| `cvfm4rehab_longva` | longva_7b |

NOTE: longvila_8b, longva_7b not tested yet, longva_7b

**Reproducing the Results**

```bash

mv evaluate.sh.example evaluate.sh  # MAKE SURE TO fill in appropriate API keys
bash evaluate.sh --model all --task strokerehab_summarization,strokerehab_primitives

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
