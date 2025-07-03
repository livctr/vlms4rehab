from typing import Any, Tuple, Optional

import cv2
import numpy as np

from vidplot.core import Renderer, DataStreamer


class RGBRenderer(Renderer):
    """
    Renderer for displaying RGB/BGR/grayscale frames with flexible resizing and background options.
    Args:
        name: Unique name for the renderer.
        data_streamer: The data streamer providing frames.
        grid_row: Tuple of (start_row, end_row) in grid.
        grid_column: Tuple of (start_col, end_col) in grid.
        channel: 'rgb' or 'bgr'. Determines expected channel order for output.
        resize_mode: 'fit' (keep aspect, pad), 'stretch' (fill bbox), or 'center'
            (no resize, center in bbox).
        background: Background color as tuple (R,G,B) or (R,G,B,A) or None for transparent.
        z_index: Depth ordering; larger values drawn on top.
    """

    def __init__(
        self,
        name: str,
        data_streamer: DataStreamer,
        grid_row: Tuple[int, int],
        grid_column: Tuple[int, int],
        channel: str = "rgb",
        resize_mode: str = "fit",
        background: Optional[Tuple[int, ...]] = (0, 0, 0),
        z_index: int = 0,
    ):
        super().__init__(name, data_streamer, grid_row, grid_column, z_index=z_index)
        assert channel in ("rgb", "bgr"), "channel must be 'rgb' or 'bgr'"
        assert resize_mode in (
            "fit",
            "stretch",
            "center",
        ), "resize_mode must be 'fit', 'stretch', or 'center'"
        self.channel = channel
        self.resize_mode = resize_mode
        self.background = background

    def _default_size(self) -> Tuple[int, int]:
        return (100, 100)

    def _convert_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:  # Grayscale
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if self.channel == "rgb":
            return frame
        elif self.channel == "bgr":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _render(self, data: Any, bbox: Tuple[int, int, int, int], canvas: Any) -> Any:
        x1, y1, x2, y2 = bbox
        target_w, target_h = x2 - x1, y2 - y1
        if data is None:
            return canvas
        frame = np.array(data)
        frame = self._convert_to_rgb(frame)
        fh, fw = frame.shape[:2]
        # Prepare background
        if self.background is None:
            bg = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            bg = np.ones((target_h, target_w, 3), dtype=np.uint8)
            for c in range(3):
                bg[..., c] *= self.background[c] if c < len(self.background) else 0
        # Resize logic
        if self.resize_mode == "stretch":
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            bg[:, :, :] = resized
        elif self.resize_mode == "fit":
            scale = min(target_w / fw, target_h / fh)
            new_w, new_h = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            y_off = (target_h - new_h) // 2
            x_off = (target_w - new_w) // 2
            bg[y_off : y_off + new_h, x_off : x_off + new_w, :] = resized
        elif self.resize_mode == "center":
            y_off = (target_h - fh) // 2
            x_off = (target_w - fw) // 2
            if 0 <= y_off < target_h and 0 <= x_off < target_w:
                h_clip = min(fh, target_h - y_off)
                w_clip = min(fw, target_w - x_off)
                bg[y_off : y_off + h_clip, x_off : x_off + w_clip, :] = frame[:h_clip, :w_clip, :]
        # Place on canvas
        canvas[y1:y2, x1:x2, :] = bg
        return canvas
