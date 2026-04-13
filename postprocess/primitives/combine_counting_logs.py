"""
Combine the original counting JSONL with partition results into a single file.

Usage:
    python postprocess/primitives/combine_counting_logs.py [--out PATH]

By default writes to:
    logs/strokerehab_counting/qwen2_5_vl_prim/combined_samples_strokerehab_counting.jsonl
"""
import argparse
import glob
import os
from typing import List


ORIGINAL_SAMPLES = (
    "logs/strokerehab_counting/qwen2_5_vl_prim/"
    "20251031_224334_samples_strokerehab_counting.jsonl"
)
PARTS_GLOB = (
    "logs/counting_parts/part_*/strokerehab_counting/qwen2_5_vl_prim/"
    "*_samples_strokerehab_counting.jsonl"
)
DEFAULT_OUT = (
    "logs/strokerehab_counting/qwen2_5_vl_prim/"
    "combined_samples_strokerehab_counting.jsonl"
)


def find_part_files() -> List[str]:
    found = sorted(glob.glob(PARTS_GLOB))
    return found


def combine(sources: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = 0
    with open(out_path, "w") as fout:
        for src in sources:
            with open(src) as fin:
                lines = [l for l in fin if l.strip()]
            fout.writelines(lines)
            total += len(lines)
            print(f"  {len(lines):4d} samples  <- {src}")
    print(f"\nTotal: {total} samples -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--parts-glob",
        default=PARTS_GLOB,
        help="Glob for partition sample files (default: %(default)s)",
    )
    args = parser.parse_args()

    part_files = sorted(glob.glob(args.parts_glob))

    if not os.path.exists(ORIGINAL_SAMPLES):
        raise FileNotFoundError(f"Original samples not found: {ORIGINAL_SAMPLES}")
    if not part_files:
        raise FileNotFoundError(
            f"No partition files matched: {args.parts_glob}\n"
            "Have the sbatch jobs finished?"
        )

    sources = [ORIGINAL_SAMPLES] + part_files
    print(f"Combining {len(sources)} files:")
    combine(sources, args.out)


if __name__ == "__main__":
    main()
