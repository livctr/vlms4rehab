"""
This module defines the Aligner class, an interface for traversing multiple
DataIterator objects.

Aligns 2+ data sequences with possibly different sampling rates.
"""

from __future__ import annotations
from typing import Iterator, List, Optional, Tuple

from .data_iterator import DataIterator


class Aligner:
    """
    :attr metadata: List of dictionary metadata. Each dictionary contains metadata
        from the iterator at the same index.
    """
    def __init__(self, iterators: List[DataIterator],
                 start_offsets: List[int] | List[float],
                 mode: str = "time") -> None:
        """
        :param iterators: List of DataIterator objects to align, each yielding
            (timestamp, data).
        :param start_offsets: List of relative time offsets for each iterator. Every
            value must be non-negative.
        """
        if len(iterators) == 0:
            raise ValueError("At least one iterator must be provided.")
        for iterator in iterators:
            if mode == "index" and iterator.mode != "index":
                raise ValueError("All iterators must be in index mode.")
            if mode == "time" and iterator.mode != "time":
                raise ValueError("All iterators must be in time mode.")
        if start_offsets is not None:
            for offset in start_offsets:
                if offset < 0:
                    raise ValueError("All start offsets must be non-negative.")
        if start_offsets is not None:
            if len(start_offsets) != len(iterators):
                raise ValueError("The number of start offsets must match the number of iterators.")

        self._iterators = iterators
        zero = 0 if mode == "index" else 0.0
        self._start_offsets = [zero] * len(iterators) if start_offsets is None else start_offsets
        self.metadata = [iterator.metadata for iterator in iterators]

        # Internal timekeeping
        self._ts = zero

    def __iter__(self) -> Aligner:
        return self
    
    def __next__(self) -> Tuple[any]:
        """
        Return the next aligned data from all iterators.

        :return: A tuple containing the data from each iterator.
        """
        pass


class IndexAligner(Aligner):
    def __init__(
        self,
        iterators: List[DataIterator],
        start_offsets: Optional[List[int]] = None,
    ) -> None:
        """
        :param iterators: List of DataIterator objects to align, each yielding
            (timestamp, data). The DataIterator mode must be `index`.
        """
        super().__init__(iterators, start_offsets, mode="index")

    def __next__(self) -> Tuple[int, List[any]]:
        """
        Return the next aligned indices and data from all iterators.

        :return: A tuple containing the index and a list of data from each iterator.
        """
        data_list = [None for _ in range(len(self._iterators))]
        num_finished = 0

        for i, iterator in enumerate(self._iterators):
            if self._ts < self._start_offsets[i]:
                continue

            n = next(iterator, None)
            if n is None:
                num_finished += 1
                continue
            _, data = n
            data_list[i] = data
        
        if num_finished == len(self._iterators):
            raise StopIteration

        cur_ts = self._ts
        self._ts += 1
        return cur_ts, tuple(data_list)


class TimeAligner(Aligner):
    """
    :attr sample_rate: Sampling rate for alignment.
    """
    def __init__(
        self, 
        iterators: List[DataIterator],
        start_offsets: Optional[List[float]] = None,
        sample_rate: Optional[float] = None,
    ) -> None:
        """
        :param iterators: List of DataIterator objects to align, each yielding
            (timestamp, data). The DataIterator mode must be `time`.
        :param sample_rate: Optional sampling rate for alignment. If set, use 
            this sample rate. Otherwise, use first iterator's sample rate.
        :param start_offsets: List of relative time offsets for each iterator. Every
            value must be non-negative.
        """
        super().__init__(iterators, start_offsets, mode="time")
        self.sample_rate = sample_rate if sample_rate is not None else iterators[0].sample_rate

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate: float) -> None:
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")
        self._sample_rate = sample_rate
        # synchronize the sample rate across iterators
        for iterator in self._iterators:
            iterator.sample_rate = sample_rate

    def __next__(self) -> Tuple[float, List[any]]:
        """
        Return the next aligned timestamps and data from all iterators.

        :return: A tuple containing the timestamp and a list of data from each iterator.
        """
        data_list = [None for _ in range(len(self._iterators))]
        num_finished = 0

        for i, iterator in enumerate(self._iterators):
            if self._ts < self._start_offsets[i]:
                continue

            n = next(iterator, None)
            if n is None:
                num_finished += 1
                continue
            _, data = n
            data_list[i] = data

        if num_finished == len(self._iterators):
            raise StopIteration

        cur_ts = self._ts
        self._ts += self._sample_rate
        return cur_ts, tuple(data_list)
