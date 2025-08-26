from collections import defaultdict
import json
import subprocess
from pathlib import Path

import pandas as pd

from data.utils_strokerehab import FM_ITEM_TO_FM_RANGE, DataPaths


def extract_patient(path_v: str) -> str:
    if '/' in path_v:
        return path_v.split('/', 1)[0]
    return path_v.split('_', 1)[0]

def add_folder_if_needed(path_v: str) -> str:
    if '/' in path_v:
        return path_v
    folder = path_v.split('_', 1)[0]
    return f"{folder}/{path_v}"

def get_video_meta(path: Path):
    """Return (width, height, duration_seconds) via ffprobe JSON."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration",
        "-of", "json", str(path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    s = info["streams"][0]
    return int(s["width"]), int(s["height"]), float(s["duration"])

def concat_pair_clip(
    left_vid: Path,
    right_vid: Path,
    out_dir: Path,
    patient: str,
    fm_range: str,
    view: str
):
    """Concatenate a left/right pair into B[T/S], padding shorter video with black."""
    wL, hL, dL = get_video_meta(left_vid)
    wR, hR, dR = get_video_meta(right_vid)

    # decide mode by comparing the shorter dimension
    short_w, short_h = min(wL, wR), min(hL, hR)
    if short_w < short_h:
        # width is shorter → side-by-side (S)
        label = "S"
        # pad both to max height
        max_h = max(hL, hR)
        pad0 = f"[0:v]pad=iw:{max_h}:(ow-iw)/2:0:color=black[v0];"
        pad1 = f"[1:v]pad=iw:{max_h}:(ow-iw)/2:0:color=black[v1];"
        stack = "[v0][v1]hstack=inputs=2[v]"
    else:
        # height is shorter → top/bottom (T)
        label = "T"
        max_w = max(wL, wR)
        pad0 = f"[0:v]pad={max_w}:ih:(ow-iw)/2:0:color=black[v0];"
        pad1 = f"[1:v]pad={max_w}:ih:(ow-iw)/2:0:color=black[v1];"
        stack = "[v0][v1]vstack=inputs=2[v]"

    out_name = f"{patient}_FM{fm_range}_B{label}_{view}.mp4"
    out_path = out_dir / out_name
    if out_path.exists():
        # print(f"✅  Concatenated exists: {out_path}, skipping.")
        return

    filter_complex = pad0 + pad1 + stack
    cmd = [
        "ffmpeg",
        "-i", str(left_vid),
        "-i", str(right_vid),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(out_path)
    ]
    print(f"🔗  Concatenating → {out_path.name}")
    subprocess.run(cmd, check=True)


def get_patient_to_affected_side_mapping():
    """
    Map patient IDs to their affected side. For control patients, we default to "Right".
    """
    df = pd.read_csv(DataPaths.IA_SCORES_PATH, dtype={'Subject ID': str})
    mapping = {}
    for _, row in df.iterrows():
        patient = row['Subject ID']
        affected_side = row['Side of body affected']
        if pd.isna(affected_side):
            mapping[patient] = "Right"
        else:
            mapping[patient] = affected_side
    return mapping



# FM_ITEM_TO_FM_RANGE = {
#     3: (3, 8), 4: (3, 8), 5: (3, 8), 6: (3, 8), 7: (3, 8), 8: (3, 8),
#     9: (9, 11), 10: (9, 11), 11: (9, 11),
#     12: (12, 12), 13: (13, 13), 14: (14, 14), 15: (15, 15), 16: (16, 16), 17: (17, 17),
#     18: (18, 18), 19: (19, 19), 20: (20, 20), 21: (21, 21), 22: (22, 22), 23: (23, 23),
#     24: (24, 25), 25: (24, 25),
#     26: (26, 26), 27: (27, 27), 28: (28, 28), 29: (29, 29), 30: (30, 30),
#     31: (31, 33), 32: (31, 33), 33: (31, 33),
# }

def extract_FM_clips(filter_by_patients=None, force_extract=False):
    """
    Extract the FM clips using the manual temporal annotations.

    Arguments:
    - filter_by_patients: List of patient IDs to filter the clips (default: None)
    - force_extract: If True, force extraction even if clips already exist (default: False)
    """

    # Get the list of clips and their times
    df = pd.read_csv(DataPaths.IA_CLIPS_PATH, sep=';')
    df['patient'] = df['video_path'].apply(extract_patient)
    if filter_by_patients is not None:
        df = df[df['patient'].isin(filter_by_patients)].copy()
    df['video_path'] = df['video_path'].apply(add_folder_if_needed)
    df['fm_range']  = df['fm_item'].str[:-1].str.replace('-', '_')
    df['side']      = df['fm_item'].str[-1]  # Either 'L' or 'R'

    # We use both views
    def swap_1_2(path: str) -> str:
        if "1.mp4" in path:
            return path.replace("1.mp4", "2.mp4")
        elif "2.mp4" in path:
            return path.replace("2.mp4", "1.mp4")
        return path
    df_swapped = df.copy()
    df_swapped['video_path'] = df_swapped['video_path'].apply(swap_1_2)
    df = pd.concat([df, df_swapped], ignore_index=True)
    df.sort_values("video_path", inplace=True)

    # Infuse view and affected side data
    vdf = pd.read_csv(DataPaths.VIEWS_PATH)
    df = df.merge(vdf, left_on="video_path", right_on="path_v", how="left").drop(columns=['path_v'])
    affected_side_mapping = get_patient_to_affected_side_mapping()

    # extract individual clips & record their paths
    clips_map = {}  # (patient,fm_range,view) → {'H': [...], 'A': [...]}  # healthy, affected
    for _, row in df.iterrows():
        patient       = row['patient']
        video_path    = row['video_path']
        fm_range      = row['fm_range']
        video_side    = row['side']  # either 'L' or 'R'
        times_str     = row['times']
        view          = row['view']  # either 'Front', 'Right', or 'Left'
        affected_side = affected_side_mapping.get(patient)  # either 'Left' or 'Right'

        input_path = Path(DataPaths.IA_RAW_VIDEO_DIR) / video_path
        if not input_path.exists():
            print(f"❌  Input not found: {input_path}")
            continue

        parts = [p.strip() for p in times_str.split(',')]
        if len(parts) % 2 != 0:
            print(f"⚠️  Odd tokens, skipping: {times_str}")
            continue

        out_dir = Path(DataPaths.IA_CLIPPED_VIDEO_DIR) / patient
        out_dir.mkdir(parents=True, exist_ok=True)

        idx = 0
        segs = []
        start, end = None, None
        while idx < len(parts):
            annot = parts[idx].split(':',1)
            if annot[0] == 's':
                start = annot[1]
            elif annot[0] == 'e':
                end = annot[1]
                if start is None:
                    raise ValueError(f"Missing start time for segment: {times_str}")
                segs.append((start, end))
                start, end = None, None
            idx += 1
        
        # Rule-based extraction
        # - If the FM item has one clip, extract it.
        # - Otherwise, extract the second clip.
        if len(segs) == 1:
            start, end = segs[0]
        elif len(segs) > 1:
            start, end = segs[1]
        else:
            raise ValueError(f"Missing segment for times: {times_str}. Video: {video_path}. FM item: {fm_range}")

        # `laterality` = which arm we are watching? The affected ('A') or healthy ('H') one?
        if ((video_side == 'L') and (affected_side == 'Left')) or \
           ((video_side == 'R') and (affected_side == 'Right')):
            laterality = "A"  # We're watching the affected side
        elif ((video_side == 'L') and (affected_side == 'Right')) or \
             ((video_side == 'R') and (affected_side == 'Left')):
            laterality = "H"  # We're watching the healthy side
        else:
            raise ValueError(f"Unknown laterality: {video_side}, {affected_side}")

        # `view` = which angle are we watching from?
        # - F = front view
        # - S = the view on the affected side
        # - S = the view on the healthy side (set as S for consistency during concatenation)
        if view == 'Front':
            view = 'F'
        elif ((view == 'Right') and (affected_side == 'Right')) or \
             ((view == 'Left') and (affected_side == 'Left')):
            view = 'S'
        elif ((view == 'Right') and (affected_side == 'Left')) or \
             ((view == 'Left') and (affected_side == 'Right')):
            view = 'S'
            # Given data collection, we don't expect this
            print(f"⚠️  Unexpected healthy-side view for patient {patient}, video {video_path}")
        else:
            raise ValueError(f"Unknown view: {view}, {affected_side}")
    
        # Save the video
        out_name = f"{patient}_FM{fm_range}_{laterality}_{view}.mp4"
        out_path = out_dir / out_name
        # record for later concatenation
        clips_map.setdefault((patient, fm_range, view), {}).setdefault(laterality, []).append(out_path)
        if out_path.exists() and not force_extract:
            print(f"✅  Clip exists: {out_path}, skipping.")
            continue
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-ss", start, "-to", end,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(out_path)
        ]
        print(f"🎬  Extracting {idx:02d}: {start} → {end} → {out_path.name}")
        subprocess.run(cmd, check=True)

    # For each fm_range, pair up H/A and produce concatenations
    for (patient, fm_range, view), sides in clips_map.items():
        A_list = sorted(sides.get('A', []))
        H_list = sorted(sides.get('H', []))
        num = min(len(A_list), len(H_list))
        out_dir = Path(DataPaths.IA_CLIPPED_VIDEO_DIR) / patient
        if num > 0:
            concat_pair_clip(A_list[0], H_list[0], out_dir, patient, fm_range, view)


def write_ia_video_metadata(
    output_path=DataPaths.IA_VIDEO_METADATA_PATH1,
    questions_path=DataPaths.IA_QUESTIONS_PATH1,
    use_view="Question"
):
    """
    Write the set of videos with attached questions to the metadata file.

    Arguments:
    - output_path: Path to the output metadata file.
    - questions_path: Path to the questions CSV file.

    CSV Header: "path_v,patient,fm_low,fm_high,laterality,video_view,side_affected,annotated_view,question_view,duration"
    - path_v: Relative path to the video file.
    - patient: Patient identifier.
    - fm_low: Lowest item in Fugl-Meyer assessment for this video
    - fm_high: Highest item in Fugl-Meyer assessment for this video
    - laterality: Laterality of the view (A for affected, H for healthy).
    - video_view: "S" for side or "F" for front.
    - side_affected: "Left" or "Right"
    - annotated_view: True iff this view was the one annotated on.
    - question_view: True iff this view is referred to in the questions CSV.
    - duration: video duration in seconds.
    """
    assert use_view in ["Annotated", "Question"], f"Unknown use_view: {use_view}"

    clipped_dir = Path(DataPaths.IA_CLIPPED_VIDEO_DIR)

    # Find all .mp4 files recursively
    mp4_files = clipped_dir.rglob("*.mp4")

    # Compute relative paths
    rel_paths = [str(p.relative_to(clipped_dir)) for p in mp4_files]

    # Build metadata: path_v, patient, fm_low, fm_high, laterality, view, duration_s, side_affected
    df = pd.DataFrame(data={"path_v": rel_paths})

    def get_basename(path_v):
        if '/' in path_v:
            return path_v.rsplit('/', 1)[-1]
        return path_v

    def extract_info_from_basename(basename):
        parts = basename.split('.')[0].split('_')

        patient = parts[0]
        fm_range_part = parts[1:-2]
        if len(fm_range_part) == 1:
            fm_low = fm_high = int(fm_range_part[0][2:])
        else:
            fm_low = int(fm_range_part[0][2:])
            fm_high = int(fm_range_part[1])
        
        # f"{patient}_FM{fm_range}_{laterality}_{view}.mp4"
        laterality = parts[-2]
        view = parts[-1]

        return (
            patient, fm_low, fm_high, laterality, view
        )

    basename = df['path_v'].apply(get_basename)
    df[['patient', 'fm_low', 'fm_high', 'laterality', 'view']] = \
        basename.apply(extract_info_from_basename).apply(pd.Series)

    mapper = get_patient_to_affected_side_mapping()
    df['side_affected'] = df['patient'].map(mapper)
    # Get the view information for the patient and FM item
    annot = pd.read_csv(DataPaths.IA_CLIPS_PATH, sep=';')
    vdf = pd.read_csv(DataPaths.VIEWS_PATH)
    def remove_folder_if_needed(path):
        if '/' in path:
            return path.split('/', 1)[1]
        return path
    vdf['path_v'] = vdf['path_v'].apply(remove_folder_if_needed)

    # Get FM item information on the views
    annot = pd.merge(annot, vdf, left_on='video_path', right_on='path_v', how='left')
    annot['patient'] = annot['video_path'].apply(extract_patient)
    def extract_fm_low_high(s):
        fm_range = s[:-1].split("-")
        if len(fm_range) == 1:
            fm_low = fm_high = int(fm_range[0])
        else:
            fm_low = int(fm_range[0])
            fm_high = int(fm_range[1])
        return fm_low, fm_high
    annot[['fm_low', 'fm_high']] = annot['fm_item'].apply(extract_fm_low_high).apply(pd.Series)
    view = annot[['patient', 'fm_low', 'fm_high', 'view']].copy()
    view.rename({'view': 'is_annotated_view'}, axis=1, inplace=True)
    view = view.groupby(['patient', 'fm_low', 'fm_high']).first().reset_index()

    # Merge view information into the Dataframe
    df = pd.merge(df, view, on=['patient', 'fm_low', 'fm_high'], how='left')
    df['affected_video_loc'] = 'Center'
    df.loc[df['laterality'] == 'BT', 'affected_video_loc'] = 'Top'
    df.loc[df['laterality'] == 'BS', 'affected_video_loc'] = 'Left'
    df['laterality'] = df['laterality'].map({'BT': 'B', 'BS': 'B', 'A': 'A', 'H': 'H'})
    df.rename(columns={"view": "video_view"}, inplace=True)

    df['is_annotated_view'] = df['is_annotated_view'].apply(
        lambda x: 'F' if x == 'Front' else ('S' if x == 'Left' else ('S' if x == 'Right' else None))
    )
    df['is_annotated_view'] = df['is_annotated_view'] == df['video_view']

    # 2) Filter based on the questions asked
    questions_df = pd.read_csv(questions_path)
    questions_df['fm_item'] = questions_df['fm_video'].apply(lambda x: x.split('_')[0])
    questions_df['laterality'] = questions_df['fm_video'].apply(lambda x: x.split('_')[1])
    questions_df['is_question_view'] = questions_df['fm_video'].apply(lambda x: x.split('_')[2])
    fm_range_to_qtypes = defaultdict(set)
    for _, row in questions_df.iterrows():
        fm_range = FM_ITEM_TO_FM_RANGE[int(row['fm_item'])]
        fm_range_to_qtypes[fm_range].add((row['laterality'], row['is_question_view']))

    def is_question_view(row):
        return (row['laterality'], row['video_view']) in fm_range_to_qtypes[(row['fm_low'], row['fm_high'])]

    df['is_question_view'] = df.apply(is_question_view, axis=1)

    df['duration'] = df['path_v'].apply(lambda p: get_video_meta(clipped_dir / p)[2])

    df.sort_values(['patient', 'fm_low'], inplace=True)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    extract_FM_clips(['C00011', 'S0005', 'S0001', 'S00021'])
    write_ia_video_metadata(DataPaths.IA_VIDEO_METADATA_PATH1, DataPaths.IA_QUESTIONS_PATH1)
    write_ia_video_metadata(DataPaths.IA_VIDEO_METADATA_PATH2, DataPaths.IA_QUESTIONS_PATH2)
