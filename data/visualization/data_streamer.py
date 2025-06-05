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
import random


def _build_from_data_source(
    data_source: pd.DataFrame | str | Dict[str, Iterable],
    data_col: str,
    time_col: str
) -> Tuple[List[float], List[Any]]:

    if data_col is None:
        raise ValueError("data_col must be specified.")
    if time_col is None:
        raise ValueError("time_col must be specified if data_source is not a DataFrame with a time index.")

    # Load data based on type
    if isinstance(data_source, pd.DataFrame):
        df = data_source
    elif isinstance(data_source, str):
        data_source = Path(data_source)
        if data_source.suffix.lower() == ".csv":
            df = pd.read_csv(data_source)
        elif data_source.suffix.lower() == ".npz":
            npz = np.load(data_source, allow_pickle=True)
            df = pd.DataFrame({key: npz[key] for key in npz.files if key in [data_col, time_col]})
        else:
            raise ValueError(f"Unsupported file format: {data_source.suffix}")
    elif isinstance(data_source, dict):
        df = pd.DataFrame({key: pd.Series(value) for key, value in data_source.items() if key in [data_col, time_col]})
    else:
        raise TypeError("Unsupported data_source type.")
    if data_col not in df.columns:
        raise ValueError(f"'{data_col}' not found in columns: {df.columns}")
    if time_col and time_col not in df.columns:
        raise ValueError(f"'{time_col}' not found in columns: {df.columns}")

    timestamps = df[time_col].values
    if len(timestamps) == 0:
        raise ValueError("Timestamps found to be empty in data.")
    sorted_idx = np.argsort(timestamps)
    timestamps = timestamps[sorted_idx]
    data = df[data_col].values[sorted_idx]
    return timestamps, data


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
        self.metadata = {}
        self.time_length: float = None  # Must be set by subclass

    def __post_init__(self) -> None:
        if self.time_length is None:
            raise ValueError("Subclasses must set self.time_length in their __init__ method")

    @property
    def approx_length(self) -> Optional[int]:
        return self._approx_length

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


class DecordVideoStreamer(DataStreamer):
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
        self.time_length = len(self._video_reader) / self.video_fps
        self.height = self._video_reader[0].shape[0]
        self.width = self._video_reader[0].shape[1]

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
            self.time_length = streamer.time_length
        else:
            raise ValueError("streamer must have a valid time_length attribute set.")

    def stream(self) -> float:
        """
        Returns the current timestamp as a progress indicator.

        :return: Current timestamp in seconds.
        """
        return self._ts / self.time_length
    

class StaticHorizontalLabelBarStreamer(DataStreamer):
    """Useful for timestep-dependent labels in visualizations."""

    def __init__(self,
                 data_source: pd.DataFrame | str | Dict[str, Iterable],
                 data_col: str,
                 time_col: str,
                 color_seed: Optional[int] = None,
    ) -> None:
        """
        :param data_source: A pandas DataFrame or source that can be turned into one.
        :param data_col: The column containing label strings.
        :param time_col: Optional column indicating time (used to sort).
        :param time_length: Optional total duration (used to compute frame count).
        :param color_seed: Optional seed to keep label colors consistent.
        """
        super().__init__()
        self.timestamps, self.data = _build_from_data_source(data_source, data_col, time_col)
        self.time_length = self.timestamps[-1]
        self.color_seed = color_seed or 42
        self.label_colors = self._assign_colors(self.data)
        self.width, self.height = 400, 20  # Fixed size for the label bar
        self._label_bar = None

    def _assign_colors(self, labels: list) -> Dict[str, tuple]:
        """Assign consistent BGR colors to label strings."""
        unique_labels = sorted(set(labels))
        if len(unique_labels) > 10:
            raise ValueError("Too many unique labels for a bar plot (must be ≤ 10).")

        rng = random.Random(self.color_seed)
        colors = [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in unique_labels]
        return dict(zip(unique_labels, colors))

    def stream(self) -> np.ndarray:
        """
        Returns a (20, 400, 3) BGR image visualizing label segments over time.
        """
        if self._label_bar is None:
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            total_samples = len(self.data)
            segment_width = self.width / total_samples

            for i, label in enumerate(self.data):
                color = self.label_colors[label]
                start = int(i * segment_width)
                end = int((i + 1) * segment_width)
                image[:, start:end] = color
            
            self._label_bar = image

        return self._label_bar


class StaticStreamer(DataStreamer):
    """A static data streamer that returns a fixed value for each call."""
    def __init__(self,
                 value: Any,
                 time_length: float = float('inf')) -> None:
        """
        Initialize the StaticStreamer with a fixed value.

        :param sample_rate: The rate at which to sample data.
        :param value: The fixed value to return on each call.
        """
        super().__init__()
        self.time_length = time_length
        self.value = value

    def stream(self) -> Any:
        """
        Return the fixed value.

        :return: The fixed value.
        """
        if self.time_length is not None and self._ts > self.time_length:
            raise StopIteration("Reached time limit for static data stream.")
        return self.value


class TabularStreamer(DataStreamer):
    """A tabular data streamer.
    
    Reads all data in the beginning. The data source can be a DataFrame, a dictionary of iterables,
    or a file path to a CSV or NPZ file.
    """
    def __init__(self,
                 data_source: pd.DataFrame | str | Dict[str, Iterable],
                 data_col: str,
                 time_col: str,
    ) -> None:
        """
        Initialize TabularStreamer with the specified data source.

        :param data_source: Source of tabular data
        :param data_col: Column name containing the data to stream
        :param time_col: Column name containing timestamps, if any
        """
        super().__init__()
        self._timestamps, self._data = _build_from_data_source(data_source, data_col, time_col)
        self.time_length = self._timestamps[-1]

    def stream(self) -> Any:
        """
        Find the nearest data point based on the timestamp.

        :return: data at the specified timestamp.
        :raises StopIteration: When reaching end of data stream
        """
        idx = round(self._ts / self._sample_rate)

        if 0 <= idx < len(self._timestamps):
            while idx > 0 and \
                abs(self._timestamps[idx-1] - self._ts) < abs(self._timestamps[idx] - self._ts):
                idx -= 1
            while idx < len(self._data) - 1 and \
                abs(self._timestamps[idx+1] - self._ts) < abs(self._timestamps[idx] - self._ts):
                idx += 1

        try:
            return self._data[idx]
        except IndexError:
            if idx < 0:
                raise ValueError(f"Negative index {idx} unexpected and out of bounds.")
            raise StopIteration(f"Reached end of tabular data stream with timestamp {self._ts} and index {idx}")
