from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from vidplot.core import Renderer


class COCOKeypointsRenderer(Renderer):
    """
    Renders COCO-format keypoints on a canvas within a bounding box.
    """

    def __init__(
        self,
        name: str,
        data_streamer,
        grid_row: Tuple[int, int],
        grid_column: Tuple[int, int],
        z_index: int = 0,
        color: Tuple[int, int, int] = (0, 255, 0),  # Green in BGR
        radius: int = 3,
        thickness: int = -1,
        draw_labels: bool = False,
        keypoint_labels: Optional[Dict[int, str]] = None,
        font_scale: float = 0.4,
        font_color: Tuple[int, int, int] = (255, 255, 255),
        font_thickness: int = 1,
        font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
        confidence_threshold: float = 0.0,
        assume_normalized: Optional[bool] = None,
    ):
        """
        Parameters:
        - name: Unique name for the renderer
        - data_streamer: DataStreamer providing pose keypoints
        - grid_row: Tuple of (start_row, end_row) in grid
        - grid_column: Tuple of (start_col, end_col) in grid
        - z_index: Depth ordering; larger values drawn on top
        - color: Circle color (BGR)
        - radius: Radius of each keypoint circle
        - thickness: Thickness of the circle outline. -1 = filled
        - draw_labels: Whether to draw labels on keypoints
        - keypoint_labels: Mapping from index to string label (e.g., {4: 'LW'})
        - font_scale: Scale of font used for labels
        - font_color: Font color for labels
        - font_thickness: Thickness of label text
        - font_face: Font face for label text
        - confidence_threshold: Minimum confidence to show a keypoint
        - assume_normalized: If True/False, forces normalized/pixel input. If None, auto-detects
        """
        super().__init__(name, data_streamer, grid_row, grid_column, z_index)
        self.color = color
        self.radius = radius
        self.thickness = thickness
        self.draw_labels = draw_labels
        self.keypoint_labels = keypoint_labels or {}
        self.font_scale = font_scale
        self.font_color = font_color
        self.font_thickness = font_thickness
        self.font_face = font_face
        self.confidence_threshold = confidence_threshold
        self.assume_normalized = assume_normalized

    @property
    def _default_size(self):
        return (None, None)  # No fixed size, depends on the canvas

    def _is_normalized(self, pose: np.ndarray) -> bool:
        # Heuristic: if any keypoint is > 2, it's probably pixel-based
        if self.assume_normalized is not None:
            return self.assume_normalized
        return np.max(pose[:, :2]) <= 1.0

    def _render_pose(self, pose: np.ndarray, canvas: Any, bbox: Tuple[int, int, int, int]):
        x0, y0, w, h = bbox

        pose = np.asarray(pose)
        if pose.ndim != 2 or pose.shape[1] < 2:
            return  # Invalid pose shape

        is_norm = self._is_normalized(pose)
        keypoints = pose.copy()

        # Convert to pixel space
        if is_norm:
            keypoints[:, 0] = keypoints[:, 0] * w + x0
            keypoints[:, 1] = keypoints[:, 1] * h + y0
        else:
            keypoints[:, 0] += x0
            keypoints[:, 1] += y0

        for idx, kp in enumerate(keypoints):
            x, y = int(round(kp[0])), int(round(kp[1]))
            conf = kp[2] if kp.shape[0] > 2 else 1.0

            # Skip if confidence is too low or outside bbox
            if conf < self.confidence_threshold:
                continue
            if not (x0 <= x < x0 + w and y0 <= y < y0 + h):
                continue

            # Draw keypoint
            cv2.circle(canvas, (x, y), self.radius, self.color, self.thickness)

            # Optional label
            if self.draw_labels and idx in self.keypoint_labels:
                label = self.keypoint_labels[idx]
                cv2.putText(
                    canvas,
                    label,
                    (x + 2, y - 2),
                    self.font_face,
                    self.font_scale,
                    self.font_color,
                    self.font_thickness,
                    cv2.LINE_AA,
                )

    def _render(self, data: Any, bbox: Tuple[int, int, int, int], canvas: Any) -> Any:
        """
        Render COCO-style pose keypoints.

        Parameters:
        - data: One pose (np.ndarray of shape (K, 2 or 3)) or dict[int, pose]
        - bbox: Bounding box assumed to contain the full frame (x, y, w, h)
        - canvas: The canvas image to modify.

        Returns:
        - Modified canvas.
        """
        if data is None:
            return canvas

        if isinstance(data, dict):
            for pose in data.values():
                self._render_pose(pose, canvas, bbox)
        else:
            self._render_pose(data, canvas, bbox)

        return canvas
