import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, Optional, Tuple, Union
from vidplot.core import DataStreamer
from vidplot.streamers.utils import _load_and_validate_data_source


class LabelBarStreamer(DataStreamer):
    """
    """

    def __init__(
        self,
        name: str,
        data_source: Union[pd.DataFrame, str, Dict[str, Iterable]],
        data_col: str,
        time_col: str,
        duration: Optional[float] = None,
        num_samples: int = 1000,
        round_decimals: int = 3,
    ) -> None:
        super().__init__(name=name)

        # load raw timestamps & data
        self._timestamps, self._data = _load_and_validate_data_source(
            data_source, data_col, time_col
        )
        self._timestamps = np.asarray(self._timestamps, dtype=float)
        self._timestamps = np.round(self._timestamps, round_decimals)
        if len(self._timestamps) == 0:
            raise ValueError("No data found in the given source")

        # determine total duration
        last_ts = float(self._timestamps[-1])
        self._duration = float(duration) if duration is not None else last_ts
        assert self._duration > 0, "Duration must be positive"

        # internal pointer & flag for extra emit
        self._idx = 0
        self._emitted_extra = False

        ts_uniform = np.linspace(0.0, self._duration, num_samples, endpoint=True)
        ts_uniform = np.round(ts_uniform, round_decimals)
        data_out = []
        for t in ts_uniform:
            i = np.searchsorted(self._timestamps, t, side="right") - 1
            idx = max(0, int(i))
            data_out.append(self._data[idx])
        self._ts_uniform    = ts_uniform.tolist()     # len == num_samples
        self._data_sampled  = data_out                # len == num_samples
        self._idx           = 0

    @property
    def duration(self) -> float:
        return self._duration

    def __next__(self) -> Tuple[float, Any]:
        if self._idx >= len(self._ts_uniform):
            raise StopIteration

        ts = float(self._ts_uniform[self._idx])
        sampled_value = self._data_sampled[self._idx]
        norm = ts / self._duration

        self._idx += 1
        return ts, (self._data_sampled, sampled_value, norm)