# Contents

### Files

- `utils_strokerehab.py`: Interface for working with the data.

### fp: Functional Primitives

**Data**
- `fp/fp_metadata.csv`: See `utils_strokerehab.py` for loading the data here.
- `strokerehab_test_set.txt`: the videos in the test set of the original StrokeRehab paper. https://pubmed.ncbi.nlm.nih.gov/37766938/.

**Script(s)**
- `write_fp_metadata.py`: Script to create `fp/fp_metadata.csv`.

### ia: Impairment Assessment

**Data**
- `ANNOT_TODO.txt`: still need to annotate all these patient's IA videos.
- `fm_item_clips.csv`: the clip times in the raw FM (Fugl-Meyer assessment) videos. Manually created.
- `fm_videos.txt`: the list of paths pointing to the video clips created using `fm_item_clips.csv`. The "final" output.
- `fm_item_scores.csv`: the patient-item breakdown of scores

**Script(s)**
- `batch_convert_FM_videos.py`: converts FM videos to a more widely supported viewing format (.mp4 with a commonly used codec)
- Then, we annotate.
- `validate_annotations.py`: run this file to ensure the `fm_item_clips.csv` file is sensible.
- `extract_clips.py`: run this file to cut the .mp4 FM videos into clips.
