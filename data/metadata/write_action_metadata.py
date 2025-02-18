"""Create counts dataset from raw data."""
from collections import Counter
from functools import partial
import logging

import pandas as pd
import regex as re

from data.utils import extract_folder_file_from_path, write_metadata

from data.utils_strokerehab import DataPaths
from data.utils_strokerehab import LabelUtils


def fill_gap(path: str, fps: int = -1, verbose: bool = False):
    """Returns a list of times and actions from a csv file.

    Some files have gaps in the time. If the surrounding actions are identical,
    the gap is filled with the same action. Otherwise, it is filled with an empty string.

    If `fps` is provided, the function will check if the gap size is consistent with the fps.
    """

    df = pd.read_csv(path)
    times = df['Time_s'].tolist()
    actions = df['MarkerNames'].tolist()

    # Get time differences
    time_diff_ms = [round(1000 * (times[i+1] - times[i])) for i in range(len(times)-1)]

    # Verify gap size with provided fps
    gap_freqs = Counter(time_diff_ms).most_common(2)
    inferred_ms_gap = tuple(gf[0] for gf in gap_freqs)
    if len(inferred_ms_gap) == 1:
        inferred_ms_gap = (inferred_ms_gap[0], inferred_ms_gap[0])
    if fps != -1:
        expected_ms_gaps = (int(1000 / fps), int(1000 / fps) + 1)
        # no overlap
        if inferred_ms_gap[0] not in expected_ms_gaps and \
            inferred_ms_gap[1] not in expected_ms_gaps:
            logging.warning(f'Gap in data {inferred_ms_gap}-ms does not match expected '
                            f'gap {expected_ms_gaps} from fps={fps} for {path}.')

    gap_idxs = []
    for i, t in enumerate(time_diff_ms):
        if t != inferred_ms_gap[0] and t != inferred_ms_gap[1]:
            gap_idxs.append(i)
            if verbose:
                gap_print_out = list(zip(times[i-1:i+3], actions[i-1:i+3])) \
                    if i > 1 and i+3 <= len(times) else ""
                logging.warning(f'Filled unexpected gap {t}-ms (fps={fps}) between frames '
                                f'at index {i} and {i+1} in {path}: {gap_print_out}.')

    if verbose and len(gap_idxs) > 3:
        logging.warning(f'Found {len(gap_idxs)} unexpected gaps in the file. Please manually check {path}.')

    for i in gap_idxs[::-1]:
        times.insert(i+1, round(times[i] + 1 / fps, 3))
        if actions[i] == actions[i+1]:
            actions.insert(i+1, actions[i])
        else:
            actions.insert(i+1, '')

    return times, actions


def get_label_info(path: str, fps: int = -1, verbose: bool = False):
    """Get csv label info. Use nonnegative fps to verify time gaps."""
    times, _ = fill_gap(path, fps, verbose)

    folder, file = extract_folder_file_from_path(path)
    patient = folder
    stroke = folder[0] == 'S'
    activity = re.sub(r'\d', '', file.split('_')[1])

    return {
        "path": path, "tstart": times[0], "tend": times[-1], "nlabels": len(times),
        "patient": patient, "stroke": stroke, "activity": activity,
    }


def write_label_metadata(**kwargs):
    """
    Writes video metadata in folder `DataPaths.RAW_LABEL_DIR` to
    `DataPaths.LABEL_METADATA_PATH`. See `get_label_info` for kwargs.
    """
    fn = partial(get_label_info, **kwargs)
    write_metadata(DataPaths.RAW_LABEL_DIR, DataPaths.LABEL_METADATA_PATH, fn)
    print(f"Wrote metadata to {DataPaths.LABEL_METADATA_PATH}.")
