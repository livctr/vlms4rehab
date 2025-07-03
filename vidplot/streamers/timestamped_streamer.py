import pandas as pd
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from vidplot.core import DataStreamer
from vidplot.streamers.utils import _load_and_validate_data_source


class TimestampedDataStreamer(DataStreamer):
    """
    A tabular data streamer that:
      1) emits each original (timestamp, payload) in order, and then
      2) if `duration` > last timestamp, emits exactly one more
         (duration, last_payload) before stopping.

    If `duration` is omitted or ≤ last timestamp, only the raw data is emitted.
    """

    def __init__(
        self,
        name: str,
        data_source: Union[pd.DataFrame, str, Dict[str, Iterable]],
        data_col: str,
        time_col: str,
        duration: Optional[float] = None,
    ) -> None:
        super().__init__(name=name)

        # load raw timestamps & data
        self._timestamps, self._data = _load_and_validate_data_source(
            data_source, data_col, time_col
        )
        if not self._timestamps:
            raise ValueError("No data found in the given source")

        # determine total duration
        last_ts = float(self._timestamps[-1])
        self._duration = float(duration) if duration is not None else last_ts

        # internal pointer & flag for extra emit
        self._idx = 0
        self._emitted_extra = False

    @property
    def duration(self) -> float:
        return self._duration

    def __next__(self) -> Tuple[float, Any]:
        # 1) emit raw data
        if self._idx < len(self._timestamps):
            ts = float(self._timestamps[self._idx])
            payload = self._data[self._idx]
            self._idx += 1
            return ts, payload

        # 2) emit one extra (duration, last_payload) if requested
        last_ts = float(self._timestamps[-1])
        if (not self._emitted_extra) and (self._duration > last_ts):
            self._emitted_extra = True
            return self._duration, self._data[-1]

        # 3) done
        raise StopIteration
