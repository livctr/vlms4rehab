"""
This module defines the DataIterator class, which provides an iterator interface 
for traversing data based on time or sample rate specifications. It supports 
different methods for fetching data depending on whether a specific sample rate 
is provided or not.
"""
from __future__ import annotations
from typing import Optional, Tuple
from pathlib import Path

import decord
from decord import VideoReader, cpu
import numpy as np
import pandas as pd


class DataIterator:
    """Class to sequentially traverse data based on time or index.
    Implementing classes should extend the following methods.
    - sample_rate (setter)
    - _find_next_data_in_index
    - _find_next_data_in_time

    Attributes:
        sample_rate: The rate at which to sample data.
        ts_start: The start timestamp, e.g., for a video clip. If None, starts
            from the beginning.
        ts_end: The end timestamp. See ts_start. If None, goes until the end.
        sync_ts: A flag to indicate whether to synchronize timestamp during iteration.
        approx_len: length of data. Not exact if there are skips
        metadata: Metadata associated with the iterator.
    """
    def __init__(self,
                 mode: str = 'time',
                 sample_rate: Optional[float | int] = None,
                 ts_start: Optional[float | int] = None,
                 ts_end: Optional[float | int] = None,
                 sync_ts: bool = False
    ) -> None:
        """
        Initialize the DataIterator. 

        :param mode: The mode of iteration, either 'time' or 'index'.
        :param sample_rate: The rate at which to sample data, if None,
            look for next data point.
        :param ts_start: The start timestamp, e.g., for a video clip. If None,
            starts from the beginning.
        :param ts_end: The end timestamp, e.g., for a video clip. If None,
            goes until the end.
        :param sync_ts: A flag to indicate whether to synchronize timestamp
            during iteration.
        
        Note: timestamp can either be in time (e.g., seconds) or an index.
            If the mode is 'time', the timestamp should be in seconds, and if
            the mode is 'index', the timestamp should be an integer index.
        """
        if mode not in ['time', 'index']:
            raise ValueError("mode must be either 'time' or 'index'")

        self._mode = mode
        self._sample_rate = sample_rate
        self.ts_start = ts_start
        self.ts_end = ts_end
        self.sync_ts = sync_ts
        self.metadata = {}

        # Internal time-keeping mechanism
        if mode == 'time':
            self._ts = 0.0 if ts_start is None else ts_start
        else:
            self._ts = 0 if ts_start is None else ts_start
            self._sample_rate = 1 if sample_rate is None else sample_rate
    
    @property
    def mode(self):
        return self._mode

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def ts_start(self):
        return self._ts_start

    @property
    def ts_end(self):
        return self._ts_end

    @property
    def sync_ts(self):
        return self._sync_ts

    @sample_rate.setter
    def sample_rate(self, value):
        """Allows sample rate to be set after init for the iterator."""
        if self._mode == 'time':
            if not isinstance(value, float):
                raise ValueError("sample_rate must be a float in time mode")
        else:
            if not isinstance(value, int):
                raise ValueError("sample_rate must be an integer in index mode")
        self._sample_rate = value

    @ts_start.setter
    def ts_start(self, value):
        """Allows ts_start to be set after init for the iterator."""
        if value is None:
            self._ts_start = value
            return
        if self._mode == 'time':
            if value is not None and not isinstance(value, float):
                raise ValueError("ts_start must be a float in time mode")
        else:
            if value is not None and not isinstance(value, int):
                raise ValueError("ts_start must be an integer in index mode")
        self._ts_start = value   

    @ts_end.setter
    def ts_end(self, value):
        """Allows ts_end to be set after init for the iterator."""
        if self._mode == 'time':
            if value is not None and not isinstance(value, float):
                raise ValueError("ts_end must be a float in time mode")
        else:
            if value is not None and not isinstance(value, int):
                raise ValueError("ts_end must be an integer in index mode")
        self._ts_end = value  

    @sync_ts.setter
    def sync_ts(self, value):
        """Allows sync_ts to be set after init for the iterator."""
        if not isinstance(value, bool):
            raise ValueError("sync_ts must be a boolean")
        self._sync_ts = value  

    def _raise_stop_iteration_if_exceed_end(self) -> None:
        """
        Raise a StopIteration exception if the iteration has passed the end time.
        """
        if self._ts_end is not None and self._ts > self._ts_end:
            raise StopIteration

    def _find_next_data_in_index(self):
        """
        Find the next data point based on the index. Implementing classes are
        expected to override this method.

        :raises StopIteration: If the iteration has passed the end time.
        :return: A tuple of (timestamp, data) at the next index.
        """
        pass
    
    def _find_next_data_in_time(self):
        """
        Find the nearest data point based on the timestamp. Implementing classes are
        expected to override this method by setting _closest
        and returning it.

        :raises StopIteration: If the iteration has passed the end time.
        :return: A tuple of (timestamp, data) at the nearest timestamp.
        """
        pass

    def __iter__(self) -> DataIterator:
        """
        Return self to allow iteration over this instance.

        :return: The iterator object (self).
        """
        return self

    def __next__(self) -> Tuple[float, any]:
        """
        Retrieve the next item in the sequence based on the sample_rate setting.
        Ensure all resources are released when the iteration is complete.

        :raises ValueError: If sample_rate is neither None nor a float.
        :raises StopIteration: When there are no more items to return.
        :return: The next data item or result of _find_next_data_in_index/_find_next_data_in_time.
        """
        # Ensure sample rate is set
        if self._sample_rate is None:
            raise ValueError("sample_rate must be set")

        if self._mode == 'time':
            return self._find_next_data_in_time()
        else:
            return self._find_next_data_in_index()

                 
class DecordVideoIterator(DataIterator):
    def __init__(self,
                 path_or_video_reader: str | Path | decord.VideoReader,
                 read_from_cpu_id: int = 0,
                 **kwargs
    ) -> None:
        """
        Initialize the DecordVideoIterator.

        :param path_or_video_reader: Either a path to the video file or an already 
                                     initialized Decord video reader object.
        :param read_from_cpu_id: The CPU ID from which to read the video. Ignored
                                    if path_or_video_reader is a Decord video reader.
        :param sample_rate: if None, overrides the default sample rate to 1/fps.
        :param kwargs: Additional keyword arguments to pass to the superclass constructor.

        Example usage:
        video_path = "<path_to_video>"
        video_iterator = DecordVideoIterator(video_path, sample_rate = None)  # can set too
        for ts, data in video_iterator:
            print(f"Timestamp: {ts}, Data: {data.shape}")
        """
        super().__init__(**kwargs)

        if isinstance(path_or_video_reader, str) or isinstance(path_or_video_reader, Path):
            self.video_reader = decord.VideoReader(path_or_video_reader,
                                                   ctx=cpu(read_from_cpu_id))
        else:
            self.video_reader = path_or_video_reader
        if len(self.video_reader) == 0:
            raise ValueError("VideoReader is empty")

        self._fps = self.video_reader.get_avg_fps()

        # Iterator sample rate, can be set and different from 1/fps
        self.sample_rate = 1.0 / self._fps if self.sample_rate is None else self.sample_rate

        self.metadata.update({"length": len(self.video_reader), "fps": self._fps,
                              "height": self.video_reader[0].shape[0], "width": self.video_reader[0].shape[1]})

        self.approx_len = len(self.video_reader)

    def _find_next_data_in_index(self) -> Tuple[float, np.ndarray]:
        """
        Find the next data point based on the index.

        :param ts: The timestamp at which to retrieve the data point.
        :return: data at the specified timestamp.
        """
        self._raise_stop_iteration_if_exceed_end()
        if self._ts >= len(self.video_reader):
            raise StopIteration

        # Retrieve data at the current timestamp.
        cur_ts = self._ts
        cur_data = self.video_reader[self._ts].asnumpy()
        self._ts += self.sample_rate
        return cur_ts, cur_data

    def _find_next_data_in_time(self) -> Tuple[float, np.ndarray]:
        """
        Find the nearest data point based on the timestamp.

        :param ts: The timestamp at which to retrieve the data point.
        :return: data at the specified timestamp.
        """
        self._raise_stop_iteration_if_exceed_end()

        # implementation here is easy b/c decord allows random access, and
        # we can directly calculate the index based on the timestamp
        video_idx = round(self._ts / self.sample_rate)
        if video_idx >= len(self.video_reader):
            raise StopIteration
        cur_ts = self._ts
        cur_data = self.video_reader[video_idx].asnumpy()
        self._ts += self.sample_rate
        if self._sync_ts:
            self._ts = (video_idx + 1) * self.sample_rate
        return cur_ts, cur_data


class PandasIterator(DataIterator):
    def __init__(self,
                 df_or_path: pd.DataFrame | str | Path,
                 data_col: str,
                 time_col: str = None,
                 **kwargs
    ) -> None:
        """
        Initialize the PandasIterator.

        :param df_or_path: Either a pandas DataFrame or a path to a CSV file. The DataFrame
            should have one column that contains the time series data.
        :param data_col: The name of the column that contains the data.
        :param time_col: The name of the column that contains the timestamps. If None,
                          the index of the DataFrame is used.

        Example usage:
        csv_path = "<path_to_csv>"
        pandas_iterator = PandasIterator(csv_path, data_col="MarkerNames",
                                         time_col="Time_s", mode='time', sample_rate=1/60)
        for ts, data in pandas_iterator:
            print(f"Timestamp: {ts}, Data: {data}")
        """
        super().__init__(**kwargs)

        if isinstance(df_or_path, pd.DataFrame):
            df = df_or_path
        elif isinstance(df_or_path, str) or isinstance(df_or_path, Path):
            df = pd.read_csv(df_or_path)
        else:
            raise ValueError("df_or_path must be a pandas DataFrame or a path to a CSV file")
        
        if data_col not in df.columns:
            raise ValueError(f"Data column '{data_col}' not found in DataFrame columns: {df.columns.values}")
        if time_col is not None:
            if time_col not in df.columns:
                raise ValueError(f"Timestamp column '{time_col}' not found in DataFrame columns: {df.columns.values}")
            if self._mode == 'index':
                raise ValueError("time_col should not be set in index mode")

        # Ensure data is sorted
        self._timestamps = df[time_col].values if time_col is not None else df.index.values
        sorted_idx = np.argsort(self._timestamps)
        self._timestamps = self._timestamps[sorted_idx]
        self._data = df[data_col].values[sorted_idx]
        assert len(self._timestamps) >= 2, "Data must have length >= 2"

        self._idx = None
        # Rows per second, similar to fps in video
        self._rps = 1.0 / ((self._timestamps[-1] - self._timestamps[0]) / (len(self._timestamps) - 1))
        # Iterator sample rate, can be set and different from 1 / rps
        self.sample_rate = 1.0 / self._rps if self.sample_rate is None else self.sample_rate

        self.metadata.update({"length": len(self._timestamps), "rps": self._rps})

        self.approx_len = len(self._timestamps)

    def _find_next_data_in_index(self) -> Tuple[float, any]:
        """
        Find the next data point based on the index.

        :param ts: The timestamp at which to retrieve the data point.
        :return: data at the specified timestamp.
        """
        self._raise_stop_iteration_if_exceed_end()
        if self._ts >= len(self._data):
            raise StopIteration
        cur_ts = self._timestamps[self._ts]
        cur_data = self._data[self._ts]
        self._ts += self.sample_rate
        return cur_ts, cur_data

    def _find_next_data_in_time(self) -> Tuple[float, any]:
        """
        Find the nearest data point based on the timestamp. Handle the case
        when sometimes the DataFrame skips timestamps

        :param ts: The timestamp at which to retrieve the data point.
        :return: data at the specified timestamp.
        """
        self._raise_stop_iteration_if_exceed_end()

        # Initialize the index if not already set
        if self._idx is None:
            # Guess the index and clamp it
            self._idx = round(self._ts / self.sample_rate)
            self._idx = max(0, min(self._idx, len(self._data) - 1))
            # Seek around the initial guess
            while self._idx > 0 and \
                abs(self._timestamps[self._idx-1] - self._ts) < abs(self._timestamps[self._idx] - self._ts):
                self._idx -= 1
            while self._idx < len(self._data) - 1 and \
                abs(self._timestamps[self._idx+1] - self._ts) < abs(self._timestamps[self._idx] - self._ts):
                self._idx += 1
        
        # Done condition: index is at end and index would select timestamp outside of range
        if self._idx == len(self._data) - 1 and self._ts > self._timestamps[-1] + (self.sample_rate / 2):
            raise StopIteration

        cur_ts = self._timestamps[self._idx]
        cur_data = self._data[self._idx]
        # Update timestamp and index
        self._ts += self.sample_rate
        while self._idx < len(self._data) - 1 and \
            abs(self._timestamps[self._idx+1] - self._ts) < abs(self._timestamps[self._idx] - self._ts):
            self._idx += 1
        if self._sync_ts:
            self._ts = self._timestamps[self._idx]
        return cur_ts, cur_data



class NumpyIterator(DataIterator):
    def __init__(self,
                 df_or_path: pd.DataFrame | str | Path,
                 data_col: str,
                 ts_col: str = None,
                 **kwargs
    ) -> None:
        """
        Initialize the PandasIterator.

        :param df_or_path: Either a pandas DataFrame or a path to a CSV file. The DataFrame
            should have one column that contains the time series data.
        :param data_col: The name of the column that contains the data.
        :param ts_col: The name of the column that contains the timestamps. If None,
                          the index of the DataFrame is used.

        Example usage:
        csv_path = "<path_to_csv>"
        pandas_iterator = PandasIterator(csv_path, data_col="MarkerNames",
                                         ts_col="Time_s", mode='time', sample_rate=1/60)
        for ts, data in pandas_iterator:
            print(f"Timestamp: {ts}, Data: {data}")
        """
        assert 1 == 0, "TODO - IMPLEMENT"
        super().__init__(**kwargs)

        if isinstance(df_or_path, pd.DataFrame):
            df = df_or_path
        elif isinstance(df_or_path, str) or isinstance(df_or_path, Path):
            df = pd.read_csv(df_or_path)
        else:
            raise ValueError("df_or_path must be a pandas DataFrame or a path to a CSV file")
        
        if data_col not in df.columns:
            raise ValueError(f"Data column '{data_col}' not found in DataFrame columns: {df.columns.values}")
        if ts_col is not None and ts_col not in df.columns:
            raise ValueError(f"Timestamp column '{ts_col}' not found in DataFrame columns: {df.columns.values}")
        
        # Ensure data is sorted
        self._timestamps = df[ts_col].values if ts_col is not None else df.index.values
        sorted_idx = np.argsort(self._timestamps)
        self._timestamps = self._timestamps[sorted_idx]
        self._data = df[data_col].values[sorted_idx]
        assert len(self._timestamps) >= 2, "Data must have length >= 2"

        self._idx = None
        # Rows per second, similar to fps in video
        self._rps = 1.0 / ((self._timestamps[-1] - self._timestamps[0]) / (len(self._timestamps) - 1))
        # Iterator sample rate, can be set and different from 1 / rps
        self.sample_rate = 1.0 / self._rps if self.sample_rate is None else self.sample_rate

        self.metadata.update({"length": len(self._timestamps), "rps": self._rps})

    def _find_next_data_in_index(self) -> Tuple[float, any]:
        """
        Find the next data point based on the index.

        :param ts: The timestamp at which to retrieve the data point.
        :return: data at the specified timestamp.
        """
        self._raise_stop_iteration_if_exceed_end()
        if self._ts >= len(self._data):
            raise StopIteration
        cur_ts = self._timestamps[self._ts]
        cur_data = self._data[self._ts]
        self._ts += self.sample_rate
        return cur_ts, cur_data

    def _find_next_data_in_time(self) -> Tuple[float, any]:
        """
        Find the nearest data point based on the timestamp. Handle the case
        when sometimes the DataFrame skips timestamps

        :param ts: The timestamp at which to retrieve the data point.
        :return: data at the specified timestamp.
        """
        self._raise_stop_iteration_if_exceed_end()

        # Initialize the index if not already set
        if self._idx is None:
            # Guess the index and clamp it
            self._idx = round(self._ts / self.sample_rate)
            self._idx = max(0, min(self._idx, len(self._data) - 1))
            # Seek around the initial guess
            while self._idx > 0 and \
                abs(self._timestamps[self._idx-1] - self._ts) < abs(self._timestamps[self._idx] - self._ts):
                self._idx -= 1
            while self._idx < len(self._data) - 1 and \
                abs(self._timestamps[self._idx+1] - self._ts) < abs(self._timestamps[self._idx] - self._ts):
                self._idx += 1
        
        # Done condition: index is at end and index would select timestamp outside of range
        if self._idx == len(self._data) - 1 and self._ts > self._timestamps[-1] + (self.sample_rate / 2):
            raise StopIteration

        cur_ts = self._timestamps[self._idx]
        cur_data = self._data[self._idx]
        # Update timestamp and index
        self._ts += self.sample_rate
        while self._idx < len(self._data) - 1 and \
            abs(self._timestamps[self._idx+1] - self._ts) < abs(self._timestamps[self._idx] - self._ts):
            self._idx += 1
        if self._sync_ts:
            self._ts = self._timestamps[self._idx]
        return cur_ts, cur_data
