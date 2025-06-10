"""
This module defines the DataStreamer class, which provides an iterator interface 
for traversing data based on sample rate specifications.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import decord
import numpy as np
import pandas as pd

from data.visualization.utils import OptionalSize


def _load_and_validate_data_source(
    data_source: Union[pd.DataFrame, str, Dict[str, Any]],
    data_col: str,
    time_col: str
) -> Tuple[List[float], List[Any]]:
    """
    Loads time-series data from various formats and validates required columns.

    Args:
        data_source: Data in the form of a DataFrame, file path (CSV, NPZ, JSON), or a dict.
        data_col: Name of the column containing data values.
        time_col: Name of the column containing timestamps.

    Returns:
        A tuple of (sorted_timestamps, sorted_data), both as lists.

    Raises:
        ValueError: If columns are missing or data is empty.
        TypeError: If the data_source type is not supported.
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
            import json
            with open(path, 'r') as f:
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



def _locate_nearest_idx(ts: float, timestamps: List[float], method: str) -> int:
    """
    Locate the index of the nearest timestamp to the given timestamp `ts`.

    Args:
        ts: The target timestamp to find the nearest index for.
        timestamps: Sorted list of timestamps.
        method: Method to use for locating the nearest index. Options are:
            - "nearest": Select the nearest timestamp to the target time.
            - "nearest_left": Select the nearest timestamp that is less than or equal to the target time.
            - "nearest_right": Select the nearest timestamp that is greater than or equal to the target time.
    """
    if len(timestamps) == 1:
        return 0

    data_rate = float(timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    # Initial
    rough_idx = (ts - timestamps[0]) / data_rate
    below_idx, above_idx = int(rough_idx), int(rough_idx) + 1
    if below_idx < 0:
        below_idx, above_idx = 0, 1
    elif above_idx >= len(timestamps):
        below_idx, above_idx = len(timestamps) - 2, len(timestamps) - 1
    else:
        while below_idx > 0 and timestamps[below_idx] > ts:
            below_idx -= 1
        above_idx = below_idx + 1
        while above_idx < len(timestamps) - 1 and timestamps[above_idx] < ts:
            above_idx += 1
        below_idx = above_idx - 1

    if method == "nearest":
        idx = below_idx if abs(ts - timestamps[below_idx]) <= abs(ts - timestamps[above_idx]) else above_idx
    elif method == "nearest_left":
        idx = below_idx
    elif method == "nearest_right":
        idx = above_idx
    else:
        raise ValueError(f"Invalid stream_method: {method}. "
                            "Must be one of 'nearest', 'nearest_left', or 'nearest_right'.")
    return idx


class DataStreamer(ABC):
    """Class to sequentially traverse data based on time.

    Attributes:
        sample_rate (float): The rate at which to sample data.
        approx_length (int): Approximate length of the data stream.
        time_length (float): Total duration of the data stream in seconds. Must be set by subclasses.
    """
    def __init__(self) -> None:
        """
        Initialize the DataStreamer.
        Subclasses MUST set self.time_length in their __init__ method.
        """
        self._sample_rate: Optional[int] = None
        self._ts: float = None  # Current timestamp in seconds
        self._approx_length: Optional[int] = None

    @property
    @abstractmethod
    def time_length(self) -> float:
        """
        Abstract property: Subclasses MUST implement this property
        to return the total duration of the data stream in seconds.
        """
        pass

    @property
    def approx_length(self) -> Union[int, float]:
        return self._approx_length
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata about the video stream."""
        return {}

    @property
    def sample_rate(self) -> Optional[float]:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: Union[float, int]) -> None:
        """Subclasses should set both the sample rate and approximate length."""
        if not isinstance(value, (float, int)) or value <= 0:
            raise ValueError("sample_rate must be a positive float.")
        self._sample_rate = value
        try:
            self._approx_length = int(self.time_length / self._sample_rate)
        except OverflowError:
            self._approx_length = float('inf')

    def _update_ts(self) -> None:
        self._ts = self._ts + self._sample_rate if self._ts is not None else 0.0

    @abstractmethod
    def stream(self) -> Any:
        """
        Streams the next data point based on the current timestamp.
        
        :raises StopIteration: If the iteration has passed the end time.
        :return: A tuple of (timestamp, data) at the nearest timestamp.
        """
        pass

    def __iter__(self) -> DataStreamer:
        """Return self to allow iteration over this instance."""
        return self

    def __next__(self) -> Tuple[float, Any]:
        """Retrieve the next item in the sequence based on the sample_rate setting."""
        if self._sample_rate is None:
            raise ValueError("sample_rate must be a valid float, not None.")
        self._update_ts()
        if self._ts > self.time_length:
            raise StopIteration("Reached end of data stream")
        return self._ts, self.stream()
    

class SizedDataStreamer(DataStreamer):

    @property
    @abstractmethod
    def size(self) -> OptionalSize:
        """
        Abstract property: Subclasses of the sized data streamer
        must return the size as (width, height).
        """
        pass


class DecordVideoStreamer(SizedDataStreamer):
    """A video frame streamer using the Decord library for efficient video decoding.
    This class provides functionality to stream video frames either by index-based access
    or time-based access. It supports both file paths and pre-initialized Decord VideoReader
    objects as input.

    Attributes:
        approx_length (int): Total number of frames in the video
        video_fps (float): Average frames per second of the video
        height (int): Height of video frames in pixels
        width (int): Width of video frames in pixels
    """
    def __init__(self,
                 path_or_video_reader: str | Path | decord.VideoReader,
                 read_from_cpu_id: int = 0,
    ) -> None:
        """
        Initialize the DecordVideoStreamer which streams video frames using decord.

        :param path_or_video_reader: Either a path to the video file or an already 
            initialized Decord video reader object.
        :param read_from_cpu_id: The CPU ID from which to read the video. Ignored
            if path_or_video_reader is a Decord video reader.
        """
        super().__init__()

        if isinstance(path_or_video_reader, str) or isinstance(path_or_video_reader, Path):
            self._video_reader = decord.VideoReader(path_or_video_reader,
                                                   ctx=decord.cpu(read_from_cpu_id))
        else:
            self._video_reader = path_or_video_reader
        if len(self._video_reader) == 0:
            raise ValueError("VideoReader is empty")

        self.video_fps = self._video_reader.get_avg_fps()
        self.height = self._video_reader[0].shape[0]
        self.width = self._video_reader[0].shape[1]

    @property
    def time_length(self) -> float:
        return len(self._video_reader) / self.video_fps
    
    @property
    def size(self) -> OptionalSize:
        return (self.width, self.height)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata about the video stream."""
        return {
            "decord_version": decord.__version__,
            "fps": self.video_fps,
            "frame_width": self.width,
            "frame_height": self.height,
            "total_frames": len(self._video_reader),
            "sample_rate": self.sample_rate if self.sample_rate is not None else "None"
        }

    def stream(self) -> np.ndarray:
        """
        Get the frame closest to the current timestamp.

        :return: Tuple of (timestamp, frame) where frame is a numpy array
        :raises StopIteration: When reaching end of video
        """
        video_idx = round(self._ts * self.video_fps)
        try:
            return self._video_reader[video_idx].asnumpy()
        except IndexError:
            raise StopIteration("Reached end of video stream")


class ProgressStreamer(DataStreamer):
    """A progress streamer that returns the current fraction of time passed in the stream."""
    def __init__(self, streamer: DataStreamer) -> None:
        """
        The ProgressStreamer wraps another DataStreamer and provides a way to track
        the progress of the stream based on the total time length. The streamer must
        have a `time_length` attribute set.
        """
        super().__init__()
        if streamer is not None and type(streamer.time_length) in [int, float]:
            if streamer.time_length <= 0:
                raise ValueError("streamer.time_length must be a positive number.")
            self._streamer = streamer
        else:
            raise ValueError("streamer must have a valid time_length attribute set.")

    @property
    def time_length(self) -> float:
        return self._streamer.time_length

    def stream(self) -> float:
        """
        Returns the current timestamp as a progress indicator.

        :return: Current timestamp in seconds.
        """
        return self._ts / self.time_length


class StaticStreamer(DataStreamer):
    """A static data streamer that returns a fixed value for each call."""
    def __init__(self,
                 value: Any,
                 time_length: float = float('inf')) -> None:
        """
        Initialize the StaticStreamer with a fixed value.

        :param sample_rate: The rate at which to sample data.
        :param value: The fixed value to return on each call.
        :param time_length: The total duration of the data stream in seconds. Defaults to infinity.
        """
        super().__init__()
        self._time_length = time_length
        self.value = value

    @property
    def time_length(self) -> float:
        return self._time_length

    def stream(self) -> Any:
        """
        Return the fixed value.

        :return: The fixed value.
        """
        if self.time_length is not None and self._ts > self.time_length:
            raise StopIteration("Reached time limit for static data stream.")
        return self.value
    

class StaticTabularStreamer(DataStreamer):
    """A static tabular data streamer that returns the fixed table per call."""
    def __init__(self,
                 data_source: pd.DataFrame | str | Dict[str, Iterable],
                 data_col: str,
                 time_col: str,
                 num_samples: Optional[int] = None,
                 time_length: float = float('inf'),
                 subsample_method: str = "nearest"
    ) -> None:
        """ 
        Initialize TabularStreamer with the specified data source.

        :param data_source: Source of tabular data
        :param data_col: Column name containing the data to stream
        :param time_col: Column name containing timestamps, if any
        :param num_samples: If specified, subsample the data uniformly to this number of samples.
            Otherwise, use all data points.
        :param time_length: The total duration of the data stream in seconds. Defaults to infinity.
        :param subsample_method: Method to stream data, either "nearest", "nearest_left", or "nearest_right".
        """
        super().__init__()
        timestamps, data = _load_and_validate_data_source(data_source, data_col, time_col)

        time_samples = np.linspace(timestamps[0], timestamps[-1], num_samples) if num_samples else timestamps
        subsampled_data = []
        for time_sample in time_samples:
            idx = _locate_nearest_idx(time_sample, timestamps, subsample_method)
            subsampled_data.append(data[idx])
        self._subsampled_data = subsampled_data
        self._time_length = time_length

    @property
    def time_length(self) -> float:
        return self._time_length

    def stream(self) -> Any:
        """
        Return the first data point in the static tabular data.

        :return: The first data point.
        """
        if self.time_length is not None and self._ts > self.time_length:
            raise StopIteration("Reached time limit for static tabular data stream.")
        return self._subsampled_data  # return all data points as a list


class TabularStreamer(DataStreamer):
    """A tabular data streamer.
    
    Reads all data in the beginning. The data source can be a DataFrame, a dictionary of iterables,
    or a file path to a CSV or NPZ file.
    """
    def __init__(self,
                 data_source: pd.DataFrame | str | Dict[str, Iterable],
                 data_col: str,
                 time_col: str,
                 stream_method: str = "nearest"
    ) -> None:
        """
        Initialize TabularStreamer with the specified data source.

        :param data_source: Source of tabular data
        :param data_col: Column name containing the data to stream
        :param time_col: Column name containing timestamps, if any
        :param stream_method: Method to stream data, either "nearest", "nearest_left", or "nearest_right".
        """
        super().__init__()
        self._timestamps, self._data = _load_and_validate_data_source(data_source, data_col, time_col)
        self._stream_method = stream_method.lower()
        if self._stream_method not in ["nearest", "nearest_left", "nearest_right"]:
            raise ValueError(f"Invalid stream_method: {self._stream_method}. "
                             "Must be one of 'nearest', 'nearest_left', or 'nearest_right'.")

    @property
    def time_length(self) -> float:
        return self._timestamps[-1]

    def stream(self) -> Any:
        """
        Find the nearest data point based on the timestamp.

        :return: data at the specified timestamp.
        :raises StopIteration: When reaching end of data stream
        """
        if self._timestamps[0] <= self._ts <= self._timestamps[-1]:
            idx = _locate_nearest_idx(self._ts, self._timestamps, self._stream_method)
            return self._data[idx]
        else:
            if self._ts < 0:
                raise ValueError(f"Negative timestamp {self._ts} unexpected and out of bounds.")
            if self._ts > self._timestamps[-1]:
                raise StopIteration(f"Reached end of tabular data stream with timestamp {self._ts}")
            return None  # No data available for this timestamp
