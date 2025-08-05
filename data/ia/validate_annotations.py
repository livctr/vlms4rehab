"""
Validate finger‑movement (FM) annotation rows in a semicolon‑delimited CSV.

Usage
-----
python validate_annotations.py <csv_path> [--id ID_TO_CHECK]

If --id is omitted, the script validates *every* ID present in the file.

CSV format (semicolon separated)
--------------------------------
video_path ; fm_item ; times
S0001_FM1_1.mp4 ; 3-8L ; s:10.34,e:13.73,s:17.07,e:19.07
...
Only the `fm_item` and the subject ID (extracted from the prefix of
`video_path`, e.g. “S0001”) are used for validation – `times`
and `video_path` details are ignored.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# --- REQUIRED FM ITEMS -------------------------------------------------------
REQUIRED_ITEMS: set[str] = {
    "3-8L", "3-8R",
    "9-11L", "9-11R",
    "12L", "12R",
    "13L", "13R",
    "14L", "14R",
    "15L", "15R",
    "16L", "16R",
    "17L", "17R",
    # "18L", "18R",  # automatically scored
    "19L", "19R",
    "20L", "20R",
    "21L", "21R",
    "22L", "22R",
    "23L", "23R",
    "24-25L", "24-25R",
    "26L", "26R",
    "27L", "27R",
    "28L", "28R",
    "29L", "29R",
    "30L", "30R",
    "31-33L", "31-33R",
}

# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate FM annotation CSV.")
    p.add_argument("csv_path", type=Path, help="Path to the semicolon‑delimited CSV.")
    p.add_argument("--id", dest="single_id", help="Validate only this subject ID.")
    return p.parse_args()


def extract_subject_id(video_path: str) -> str:
    """Return the portion before the first '_' in video_path (e.g., S0001)."""
    return video_path.split("_", 1)[0]


def load_rows(csv_path: Path) -> list[tuple[str, str, int]]:
    """
    Read CSV and return a list of (subject_id, fm_item, line_number).

    line_number is 1‑based to match what a user sees in a text editor.
    """
    rows: list[tuple[str, str, int]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for idx, row in enumerate(reader, start=2):  # +2 because DictReader skips header
            subject_id = extract_subject_id(row["video_path"].strip())
            fm_item = row["fm_item"].strip()
            rows.append((subject_id, fm_item, idx))
    return rows


def validate_one(id_to_check: str, rows: list[tuple[str, str, int]]) -> list[str]:
    """
    Validate a single ID. Return a list of error strings (empty if OK).
    """
    errors: list[str] = []
    # Gather fm_items -> list[line_numbers] for this ID
    fm_to_lines: defaultdict[str, list[int]] = defaultdict(list)
    for sid, fm_item, line_no in rows:
        if sid == id_to_check:
            fm_to_lines[fm_item].append(line_no)

    if not fm_to_lines:
        errors.append(f"❌ No rows found for ID {id_to_check}.")
        return errors

    # Missing items
    missing = REQUIRED_ITEMS - fm_to_lines.keys()
    if missing:
        errors.append(f"❌ Missing items: {', '.join(sorted(missing))}")

    # Duplicates
    duplicates = {item: lines for item, lines in fm_to_lines.items() if len(lines) > 1}
    for item, lines in duplicates.items():
        pretty_lines = ", ".join(map(str, lines))
        errors.append(f"❌ Duplicate entry for {item} on CSV lines [{pretty_lines}].")

    # Unknown / extra items
    extra = set(fm_to_lines.keys()) - REQUIRED_ITEMS
    if extra:
        errors.append(f"⚠️  Unknown items (not in spec): {', '.join(sorted(extra))}")

    return errors


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv_path)

    # Decide which IDs to validate
    ids_to_check = {args.single_id} if args.single_id else {sid for sid, *_ in rows}

    overall_ok = True
    for sid in sorted(ids_to_check):
        problems = validate_one(sid, rows)
        if problems:
            overall_ok = False
            print(f"\n=== Validation for ID {sid} FAILED ===")
            for p in problems:
                print(p)
        else:
            print(f"✓ Validation for ID {sid} passed (all {len(REQUIRED_ITEMS)} items present, no duplicates).")

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
