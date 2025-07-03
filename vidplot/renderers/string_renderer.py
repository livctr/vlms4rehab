import cv2
import numpy as np
from typing import Any, Tuple, Optional
from vidplot.core import Renderer, DataStreamer


class StringRenderer(Renderer):
    """
    Renders a string within a bounding box on an image canvas using OpenCV.
    Compatible with the new Renderer API.
    """

    def __init__(
        self,
        name: str,
        data_streamer: DataStreamer,
        grid_row: Tuple[int, int],
        grid_column: Tuple[int, int],
        font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.5,
        font_color: Tuple[int, int, int] = (0, 0, 0),  # Black (BGR)
        thickness: int = 1,
        line_type: int = cv2.LINE_AA,
        num_expected_lines: int = 1,
        float_precision: Optional[int] = None,
        z_index: int = 0,
    ):
        super().__init__(name, data_streamer, grid_row, grid_column, z_index=z_index)
        self.font_face = font_face
        self.font_scale = font_scale
        self.font_color = font_color
        self.thickness = thickness
        self.line_type = line_type
        self.num_expected_lines = num_expected_lines
        self.float_precision = float_precision

    @property
    def _default_size(self) -> Tuple[Optional[int], Optional[int]]:
        (_, text_h), _ = cv2.getTextSize("test", self.font_face, self.font_scale, self.thickness)
        # Width is flexible (None), height is estimated
        return (None, text_h * self.num_expected_lines)

    def _render(self, data: Any, bbox: Tuple[int, int, int, int], canvas: np.ndarray) -> np.ndarray:
        # If data is a dictionary, extract values; else treat it as text
        if isinstance(data, dict):
            text = data.get("text", "")
            font_face = data.get("font_face", self.font_face)
            font_scale = data.get("font_scale", self.font_scale)
            font_color = data.get("font_color", self.font_color)
            thickness = data.get("thickness", self.thickness)
            line_type = data.get("line_type", self.line_type)
        else:
            text = data
            font_face = self.font_face
            font_scale = self.font_scale
            font_color = self.font_color
            thickness = self.thickness
            line_type = self.line_type

        if text is None:
            return canvas

        if isinstance(text, float) and self.float_precision is not None:
            text = f"{text:.{self.float_precision}f}"
        else:
            text = str(text)

        x, y, x2, y2 = bbox
        _, h = x2 - x, y2 - y
        # Estimate text size
        (text_w, text_h), _ = cv2.getTextSize(text, font_face, font_scale, thickness)

        # Compute text origin (top-left of text baseline), adjusting for vertical fit
        text_x = x
        text_y = y + min(h, text_h)  # draw from top of box, max height is box height

        # Ensure the text does not go outside the canvas (clip coordinates)
        text_x = max(0, min(canvas.shape[1] - text_w, text_x))
        text_y = max(text_h, min(canvas.shape[0], text_y))

        # Draw text on the image
        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            font_face,
            font_scale,
            font_color,
            thickness,
            line_type,
        )

        return canvas
