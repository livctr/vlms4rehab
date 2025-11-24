# Contents

### Files

- `utils_strokerehab.py`: Interface for working with the data. Patient severity level, paths, etc.

### fp: Functional Primitives

**Data**
- `fp/fp_metadata_description.md`: naming convention of the video paths in `fp_metadata.csv`
- `fp/fp_metadata.csv`: metadata for the video files
- `fp/strokerehab_test_set.txt`: the videos in the test set of a previous paper using the stroke rehab dataset: https://pubmed.ncbi.nlm.nih.gov/37766938/.

**Script(s)**
- `fp/write_fp_metadata.py`: Script to create `fp/fp_metadata.csv`.

### ia: Impairment Assessment

**Data**
- `ia/fm_item_clip_times.csv`: the clip times in the raw FM (Fugl-Meyer assessment) videos. Manually created.
- `ia/fm_item_questions1.csv`: question-answering, comparing paretic side to healthy side by concatenating videos side-by-side.
- `ia/fm_item_questions2.csv`: question-answering, focus only on the paretic side (used in paper).
- `ia/fm_item_questions3.csv`: chain-of-thought, comparing paretic side to healthy side by concatenating videos side-by-side.
- `ia/fm_item_questions4.csv`: chain-of-thought, focus only on the paretic side (used in paper).
- `ia/fm_item_scores.csv`: the patient-item breakdown of scores
- `ia/fm_item_views.csv`: camera position for the original, uncut FM videos.
- `ia/ia_video_metadata{i}.csv`: automatically generated from `ia/fm_item_questions{i}.csv`

**Script(s)**
- `ia/TimeTagger`: frontend for labeling temporal segments in the Fugl-Meyer assessment videos.
- `ia/scripts/batch_convert_FM_videos.py`: converts FM videos to a more widely supported viewing format (.mp4 with a commonly used codec)
- Then, we annotate.
- `ia/scripts/validate_annotations.py`: run this file to ensure the `fm_item_clips.csv` file is sensible.
- `ia/scripts/extract_clips.py`: run this file to cut the .mp4 FM videos into clips.

### rs: Functional Primitives for RTT-Shelf
- `best_views.txt`: manually selected camera stream for VLM inference
