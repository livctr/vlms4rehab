import numpy as np
import pandas as pd
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from pathlib import Path

from typing import Callable


def _stream_with_last_frame_handling(
    target_time: float,
    prev_ts: Optional[float],
    prev_frame: Any,
    cur_ts: Optional[float],
    cur_frame: Any,
    last_frame_time: Optional[float],
    last_frame: Any,
    sample_rate: float,
    seek_func: Callable[[], Tuple[float, Any]],
    selection_method: str = "nearest",
) -> Tuple[Any, float, Any, float, Any, float, Any]:
    """
    Generic streaming logic that handles last frame continuation.

    Args:
        target_time: The target time to seek to
        prev_ts: Previous timestamp
        prev_frame: Previous frame/data
        cur_ts: Current timestamp
        cur_frame: Current frame/data
        last_frame_time: Last frame timestamp (for continuation)
        last_frame: Last frame data (for continuation)
        sample_rate: Sample rate for continuation logic
        seek_func: Function to get next timestamp and frame/data
        selection_method: How to select between prev and cur
            ("nearest", "nearest_left", "nearest_right")

    Returns:
        (frame, prev_ts, prev_frame, cur_ts, cur_frame, last_frame_time, last_frame)
    """
    # initialize window on first call
    if prev_ts is None:
        # first frame
        prev_ts, prev_frame = seek_func()
        # attempt second frame; if unavailable, duplicate prev for cur
        try:
            cur_ts, cur_frame = seek_func()
        except StopIteration:
            cur_ts, cur_frame = prev_ts, prev_frame

    # advance window until cur_ts >= target_time
    while cur_ts < target_time:
        prev_ts, prev_frame = cur_ts, cur_frame
        try:
            cur_ts, cur_frame = seek_func()
        except StopIteration:
            # Video ended, but we should continue until we've processed the last frame
            # at the target sample rate
            if last_frame_time is None:
                # Cache the last frame we have
                last_frame_time = cur_ts
                last_frame = cur_frame

            # Continue rendering the last frame until target_time exceeds last_frame_time + fps
            fps = 1.0 / sample_rate
            if target_time > last_frame_time + fps:
                raise StopIteration

            # Return the last frame for any remaining time points
            return (
                last_frame,
                prev_ts,
                prev_frame,
                cur_ts,
                cur_frame,
                last_frame_time,
                last_frame,
            )

    # choose frame based on selection method
    if selection_method == "nearest":
        if abs(prev_ts - target_time) <= abs(cur_ts - target_time):
            return (
                prev_frame,
                prev_ts,
                prev_frame,
                cur_ts,
                cur_frame,
                last_frame_time,
                last_frame,
            )
        return (
            cur_frame,
            prev_ts,
            prev_frame,
            cur_ts,
            cur_frame,
            last_frame_time,
            last_frame,
        )
    elif selection_method == "nearest_left":
        return (
            prev_frame,
            prev_ts,
            prev_frame,
            cur_ts,
            cur_frame,
            last_frame_time,
            last_frame,
        )
    elif selection_method == "nearest_right":
        return (
            cur_frame,
            prev_ts,
            prev_frame,
            cur_ts,
            cur_frame,
            last_frame_time,
            last_frame,
        )
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")


def _load_and_validate_data_source(
    data_source: Union[pd.DataFrame, str, Dict[str, Any]],
    data_col: str,
    time_col: str,
) -> Tuple[List[float], List[Any]]:
    """
    Loads time-series data from various formats and validates required columns.
    """
    if not data_col:
        raise ValueError("`data_col` must be specified.")
    if not time_col:
        raise ValueError("`time_col` must be specified.")

    def sort_by_time(timestamps: Iterable, values: Iterable):
        if len(timestamps) == 0:
            raise ValueError("Data source must contain at least 1 timestamp.")
        sort_idx = np.argsort(timestamps)
        timestamps = [timestamps[i] for i in sort_idx]
        values = [values[i] for i in sort_idx]
        return timestamps, values

    if isinstance(data_source, pd.DataFrame):
        if time_col not in data_source.columns or data_col not in data_source.columns:
            raise ValueError(f"Missing required columns in DataFrame: {time_col}, {data_col}")
        return sort_by_time(data_source[time_col], data_source[data_col])

    elif isinstance(data_source, str):
        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {data_source}")

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            if time_col not in df.columns or data_col not in df.columns:
                raise ValueError(f"Missing required columns in CSV: {time_col}, {data_col}")
            return sort_by_time(df[time_col], df[data_col])

        elif path.suffix.lower() == ".npz":
            npz = np.load(path, allow_pickle=True)
            if time_col not in npz or data_col not in npz:
                raise ValueError(f"Missing keys in NPZ: {time_col}, {data_col}")
            return sort_by_time(npz[time_col], npz[data_col])

        elif path.suffix.lower() == ".json":
            with open(path, "r") as f:
                data_dict = json.load(f)
            if time_col not in data_dict or data_col not in data_dict:
                raise ValueError(f"Missing keys in JSON: {time_col}, {data_col}")
            return sort_by_time(data_dict[time_col], data_dict[data_col])

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    elif isinstance(data_source, dict):
        if time_col not in data_source or data_col not in data_source:
            raise ValueError(f"Missing keys in dict: {time_col}, {data_col}")
        return sort_by_time(data_source[time_col], data_source[data_col])

    else:
        raise TypeError(f"Unsupported data_source type: {type(data_source).__name__}")
