from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Tuple


class DataStreamer(ABC):
    """Abstract base class to sequentially traverse data based on time.

    Provides an iterable interface for streaming data points.
    Subclasses must implement the `duration` property and the `stream` method.

    Attributes:
        name (str): Name of the DataStreamer, for identification.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    @abstractmethod
    def duration(self) -> float:
        """
        Duration of the datastream in seconds. This is the total time
        over which the data is valid; this property is solely used to
        give the user an idea of approximately how long the data will be streamed.
        """
        raise NotImplementedError("Subclasses must implement duration.")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata about the data stream. Override in subclasses if needed."""
        return {}

    def __iter__(self) -> "DataStreamer":
        return self

    @abstractmethod
    def __next__(self) -> Tuple[float, Any]:
        """Retrieve the next time and item in the sequence."""
        raise NotImplementedError("Subclasses must implement __next__ method")


class SizedStreamerProtocol(Protocol):
    @property
    def size(self) -> Tuple[int, int]:
        """Return (width, height)."""
        raise NotImplementedError("Size needs to be implemented for a SizedStreamerProtocol.")


class StaticDataStreamer(DataStreamer):
    """Subclass for data streamers that always return the same data (static)."""

    def __init__(
        self,
        name: str,
        data: Any,
    ):
        super().__init__(name=name)
        self._data = data
        self._sent = False

    @property
    def duration(self) -> float:
        """Static data streams are assumed to be infinite in duration."""
        return float("inf")

    def stream(self) -> Any:
        return self._data

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "data_type": type(self._data).__name__,
            "static": True,
        }

    def __next__(self) -> Tuple[float, Any]:
        """Return the static data with a dummy timestamp."""
        if not self._sent:
            self._sent = True
            return 0.0, self._data
        else:
            raise StopIteration("StaticDataStreamer only yields data once.")
