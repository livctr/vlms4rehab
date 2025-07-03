from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional

from .streamer import DataStreamer, SizedStreamerProtocol


class Renderer(ABC):
    """Base class for rendering data from `DataStreamer`.

    This class provides the core structure for all renderers, handling
    common attributes like layout position (grid rows and columns),
    z-ordering for layering, and size awareness.

    Attributes:
        name (str): Unique identifier of the renderer.
        data_streamer (DataStreamer): Source of streaming data.
        grid_row (Tuple[int, int]): Start and end row (1-based, inclusive) in grid layout.
        grid_column (Tuple[int, int]): Start and end column (1-based, inclusive) in grid layout.
        z_index (int): Stacking order; higher values render on top.
    """

    def __init__(
        self,
        name: str,
        data_streamer: DataStreamer,
        grid_row: Tuple[int, int],
        grid_column: Tuple[int, int],
        z_index: int = 0,
    ) -> None:
        """Initialize the Renderer.

        Args:
            name: Unique name for the renderer.
            data_streamer: Provides data points to render.
            grid_row: Tuple of (start_row, end_row) in grid.
            grid_column: Tuple of (start_col, end_col) in grid.
            z_index: Depth ordering; larger values drawn on top.
        """
        self.name = name
        self.data_streamer = data_streamer
        self.grid_row = grid_row
        self.grid_column = grid_column
        self.z_index = z_index

    @property
    @abstractmethod
    def _default_size(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """
        Default (width, height) for renderers when streamer has no size.

        Subclasses must implement this to specify their preferred dimensions.
        """
        raise NotImplementedError(
            "Renderers without a sized data streamer must implement _default_size()"
        )

    @property
    def default_size(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """
        Effective size for rendering: uses streamer size if available,
        otherwise falls back to `_default_size`.
        """
        if isinstance(self.data_streamer, SizedStreamerProtocol):
            return self.data_streamer.size
        return self._default_size

    @abstractmethod
    def _render(
        self,
        data: Any,
        bbox: Tuple[int, int, int, int],
        canvas: Any,
    ) -> Any:
        """
        Draw `data` onto `canvas` within `bbox`.

        Parameters:
            data: Data to render (e.g., frames, shapes, text).
            bbox: (x, y, width, height) region on the canvas.
            canvas: Image or drawing surface to modify in place.

        Returns:
            The modified canvas.
        """
        raise NotImplementedError("Subclasses must implement _render()")

    def render(
        self,
        data: Any,
        bbox: Tuple[int, int, int, int],
        canvas: Any,
    ) -> Any:
        """
        Draw `data` onto `canvas` within `bbox`.

        Parameters:
            data: Data to render (e.g., frames, shapes, text).
            bbox: (x, y, width, height) region on the canvas.
            canvas: Image or drawing surface to modify in place.

        Returns:
            The modified canvas.
        """
        if data is None:
            return canvas
        return self._render(data, bbox, canvas)
