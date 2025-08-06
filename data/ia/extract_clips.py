import json
import subprocess
from pathlib import Path

import pandas as pd

from data.utils_strokerehab import DataPaths

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
    clip_idx: int
):
    """Concatenate a left/right pair into LR[T/S], padding shorter video with black."""
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

    out_name = f"{patient}_FM{fm_range}_LR{label}_{clip_idx:02d}.mp4"
    out_path = out_dir / out_name
    if out_path.exists():
        print(f"✅  Concatenated exists: {out_path}, skipping.")
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


def extract_clips_from_csv():
    # 1) Get the list of clips and their times
    df = pd.read_csv(DataPaths.IA_CLIPS_PATH, sep=';')
    df['patient'] = df['video_path'].apply(extract_patient)
    df['video_path'] = df['video_path'].apply(add_folder_if_needed)
    df['fm_range']  = df['fm_item'].str[:-1].str.replace('-', '_')
    df['side']      = df['fm_item'].str[-1]
    df.sort_values(['patient', 'fm_item'], inplace=True)

    # 2) extract individual clips & record their paths
    clips_map = {}  # (patient,fm_range) → {'L': [...], 'R': [...]}
    for _, row in df.iterrows():
        patient    = row['patient']
        video_path = row['video_path']
        fm_range   = row['fm_range']
        side       = row['side']
        times_str  = row['times']

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

        n_clips = len(parts) // 2
        for idx in range(n_clips):
            start = parts[2*idx].split(':',1)[1]
            end   = parts[2*idx+1].split(':',1)[1]
            out_name = f"{patient}_FM{fm_range}_{side}_{idx:02d}.mp4"
            out_path = out_dir / out_name

            # record for later concatenation
            clips_map.setdefault((patient, fm_range), {}).setdefault(side, []).append(out_path)

            if out_path.exists():
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

    # 3) for each fm_range, pair up L/R and produce concatenations
    for (patient, fm_range), sides in clips_map.items():
        L_list = sorted(sides.get('L', []))
        R_list = sorted(sides.get('R', []))
        num    = min(len(L_list), len(R_list))
        out_dir = Path(DataPaths.IA_CLIPPED_VIDEO_DIR) / patient

        for i in range(num):
            concat_pair_clip(L_list[i], R_list[i], out_dir, patient, fm_range, i)


def write_ia_video_metadata():
    """
    Write the set of videos with attached questions to the metadata file.
    """
    clipped_dir = Path(DataPaths.IA_CLIPPED_VIDEO_DIR)
    output_path = Path(DataPaths.IA_VIDEO_METADATA_PATH)

    # Find all .mp4 files recursively
    mp4_files = clipped_dir.rglob("*.mp4")

    # Compute relative paths
    rel_paths = [str(p.relative_to(clipped_dir)) for p in mp4_files]

    # 1) Build metadata: path_v, patient, fm_low, fm_high, side_shown, repetition_index, duration_s, side_affected
    df = pd.DataFrame(data={"path_v": rel_paths})

    def get_basename(path_v):
        if '/' in path_v:
            return path_v.rsplit('/', 1)[-1]
        return path_v
    
    def extract_info_from_basename(basename):
        # Example basename: S0001_FM22_R_01.mp4 / S0001_FM9_11_LRT_02.mp4
        parts = basename.split('.')[0].split('_')

        patient = parts[0]
        fm_range_part = parts[1:-2]
        if len(fm_range_part) == 1:
            fm_low = fm_high = int(fm_range_part[0][2:])
        else:
            fm_low = int(fm_range_part[0][2:])
            fm_high = int(fm_range_part[1])
        side_shown = parts[-2]  # L, R, LRT, or LRS
        repetition_index = int(parts[-1])

        return (
            patient, fm_low, fm_high, side_shown, repetition_index
        )

    basename = df['path_v'].apply(get_basename)
    df[['patient', 'fm_low', 'fm_high', 'side_shown', 'repetition_index']] = \
        basename.apply(extract_info_from_basename).apply(pd.Series)
    
    # side_affected
    df['duration_s'] = df['path_v'].apply(lambda p: get_video_meta(Path(clipped_dir) / p)[2])
    scores_df = pd.read_csv(DataPaths.IA_SCORES_PATH)
    mapper = scores_df[['Subject ID', 'Side of body affected']]
    mapper = mapper.set_index('Subject ID')['Side of body affected'].to_dict()
    df['side_affected'] = df['patient'].map(mapper)

    # 2) Filter based on the questions asked
    questions_df = pd.read_csv(DataPaths.IA_QUESTIONS_PATH)
    # `fm_videos_with_questions` is formatted as `{fm_item}_[C/I]`
    # `C` = concatenated. `I` = individual.
    # We need to see for each video, whether it is needed.
    fm_videos_with_questions = questions_df['fm_video'].unique()

    def questions_on_video_exist(row):
        fm_low = row['fm_low']
        fm_high = row['fm_high']
        side_shown = row['side_shown']
        side_affected = row['side_affected']

        seen = False
        for fm in range(fm_low + fm_high + 1):
            # First break down by what the video attempts to show
            if side_shown == 'L':  # the video shows 'L'
                if f"{fm}_I" in fm_videos_with_questions and side_affected == 'Left':
                    seen = True
                    break
            elif side_shown == 'R':  # the video shows 'R'
                if f"{fm}_I" in fm_videos_with_questions and side_affected == 'Right':
                    seen = True
                    break
            elif 'LR' in side_shown:
                if f"{fm}_C" in fm_videos_with_questions:
                    seen = True
                    break
            else:
                raise ValueError(f"Unknown side_shown: {side_shown}")
        return seen

    # Filter by questions_exist
    df = df[df.apply(questions_on_video_exist, axis=1)]

    # Sort by fm_low
    df.sort_values(['patient', 'fm_low', 'repetition_index'], inplace=True)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # extract_clips_from_csv()
    write_ia_video_metadata()
