from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from vidplot.core import Renderer
from vidplot.streamers.label_bar_streamer import LabelBarStreamer
from vidplot.renderers.string_renderer import StringRenderer


class LabelBarRenderer(Renderer):
    """Visualize time-stamp dependent labels"""

    def __init__(
        self,
        name: str,
        data_streamer: LabelBarStreamer,
        label_to_color: Dict[int, Tuple[int, int, int]],
        grid_row: Tuple[int, int],
        grid_column: Tuple[int, int],
        z_index: int = 0,
        height: int = 20,
        progress_bar_color: Tuple[int, int, int] = (0, 255, 0),
        progress_thickness: int = 2,
        write_sampled_data_str: bool = True,
    ):
        """
        Parameters:
        - name: Unique name for the renderer
        - data_streamer: DataStreamer providing label data
        - grid_row: Tuple of (start_row, end_row) in grid
        - grid_column: Tuple of (start_col, end_col) in grid
        - z_index: Depth ordering; larger values drawn on top
        - height: Height of the label bar in pixels
        - color_seed: Optional seed to keep label colors consistent
        """
        super().__init__(name, data_streamer, grid_row, grid_column, z_index)
        self._height = height
        self._label_to_color = label_to_color
        self._label_bar = None

        # a helper renderer just for drawing the sampled label text
        self._write_sampled_data_str = write_sampled_data_str
        if self._write_sampled_data_str:
            self._text_renderer = StringRenderer(
                f"{name}_text",
                data_streamer,
                grid_row,
                grid_column,
                z_index=z_index + 1  # ensure text is on top of the bar
            )

        # The tracking progress bar
        self._progress_bar_color = progress_bar_color
        self._progress_thickness = progress_thickness

    @property
    def _default_size(self):
        return (None, self._height)

    def _create_label_bar(
        self, labels: List[str], bar_height: int, bar_width: int
    ) -> Dict[str, tuple]:
        """Assign consistent BGR colors to label strings."""

        colors = [self._label_to_color[label] for label in labels]

        self._label_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

        total_samples = len(labels)
        segment_width = float(bar_width) / total_samples

        for i in range(len(labels)):
            start = int(i * segment_width)
            end = int((i + 1) * segment_width)
            self._label_bar[:, start:end] = colors[i]

    def _render(self, data: Tuple[List[Any], Any, float], bbox: Tuple[int, int, int, int], canvas: Any) -> Any:
        """
        Draw a horizontal label bar within the bounding box on the canvas.
        Each label's proportion is shown using a unique color.

        Parameters:
        - data: List of label strings
        - bbox: Bounding box (x, y, width, height) within which to draw the label bar
        - canvas: The image canvas (numpy array) to draw on

        Returns:
        - The modified canvas
        """
        if data is None:
            return canvas

        uniform_data, sampled_data, progress = data

        x1, y1, x2, y2 = bbox
        bar_width = x2 - x1
        bar_height = y2 - y1

        if bar_width <= 0 or bar_height <= 0:
            return canvas  # Nothing to draw

        # Create the label bar
        if self._label_bar is None:
            self._create_label_bar(uniform_data, bar_height, bar_width)
        canvas[y1:y2, x1:x2] = self._label_bar
        
        # Write str(sampled_data) on top of the label bar
        if self._write_sampled_data_str:
            canvas = self._text_renderer._render(
                sampled_data,
                (x1, y1, x2, y2),
                canvas
            )

        # Draw the vertical progress bar
        if not isinstance(progress, float):
            raise TypeError("LabelBarRenderer expects a float between 0.0 and 1.0.")
        if not (-0.05 <= progress <= 1.05):  # Allow a little tolerance
            raise ValueError("LabelBarRenderer: Progress value must be between 0.0 and 1.0.")
        progress = max(0.0, min(1.0, progress))  # Clamp to [0.0, 1.0]
        progress_x = x1 + int(bar_width * progress)
        half_thick = self._progress_thickness // 2
        bar_x1 = max(x1, progress_x - half_thick)
        bar_x2 = min(x2, progress_x + half_thick)
        cv2.rectangle(
            canvas,
            (bar_x1, y1),
            (bar_x2, y2),
            self._progress_bar_color,
            thickness=-1,
        )

        return canvas
