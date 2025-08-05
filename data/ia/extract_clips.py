import csv
import subprocess
from pathlib import Path

from data.utils_strokerehab import DataPaths


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_clips_from_csv(csv_path: str):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader, None)  # skip header if present

        for row in reader:
            if len(row) != 3:
                print(f"⚠️  Skipping malformed row: {row}")
                continue
            video_file, fm_item, times_str = row

            # e.g. video_file = "S0001_FM1_1.mp4"
            subject = Path(video_file).stem.split('_', 1)[0]  # e.g. "S0001"

            input_path = Path(DataPaths.IA_RAW_VIDEO_DIR) / subject / video_file
            if not input_path.exists():
                print(f"❌  Input not found: {input_path}")
                continue

            # fm_item like "3-8,L"
            try:
                fm_range_raw, side = fm_item.split(',', 1)
            except ValueError:
                print(f"⚠️  Bad fm_item format, skipping: {fm_item}")
                continue
            fm_range = fm_range_raw.replace('-', '_')  # "3-8" → "3_8"

            # parse times "s:10.34,e:13.73,s:17.07,e:19.07,…"
            parts = [p.strip() for p in times_str.split(',')]
            if len(parts) % 2 != 0:
                print(f"⚠️  Odd number of time tokens, skipping: {times_str}")
                continue

            out_dir = Path(DataPaths.IA_CLIPPED_VIDEO_DIR) / subject
            ensure_dir(out_dir)

            # every pair (s,e) is one clip
            for i in range(0, len(parts), 2):
                s_tok, e_tok = parts[i], parts[i+1]
                start = s_tok.split(':', 1)[1]
                end   = e_tok.split(':', 1)[1]
                clip_idx = i // 2

                out_name = f"{subject}_FM{fm_range}_{side}_{clip_idx:02d}.mp4"
                output_path = out_dir / out_name

                if output_path.exists():
                    print(f"✅  Clip already exists: {output_path}, skipping.")
                    continue

                cmd = [
                    "ffmpeg",
                    "-i", str(input_path),
                    "-ss", start,
                    "-to", end,
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    str(output_path)
                ]
                print(f"🎬 Extracting clip {clip_idx:02d}: {start} → {end} → {output_path}")
                subprocess.run(cmd, check=True)

if __name__ == "__main__":
    extract_clips_from_csv(DataPaths.IA_CLIPS_PATH)
