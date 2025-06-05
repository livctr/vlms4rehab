from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from data.visualization.utils import BoundingBox, Canvas, OptionalSize

import cv2
import numpy as np
from typing import Any, Tuple, Dict, Optional

from data.visualization.data_streamer import DataStreamer, DecordVideoStreamer, StaticHorizontalLabelBarStreamer


class Renderer(ABC):
    """
    Abstract base class for rendering data from a streamer.
    
    The data is rendered onto a canvas within a specified bounding box.
    """
    def __init__(self, data_streamer: DataStreamer):
        """
        Initialize the renderer with a data streamer.

        Parameters:
        - data_streamer: An instance of DataStreamer that provides the data to render.
        """
        self.data_streamer = data_streamer

    def compute_size(self) -> OptionalSize:
        """Get the size required for the data to be rendered. None means no requirements."""
        return (None, None)

    @abstractmethod
    def render(self, data: Any, bbox: BoundingBox, canvas: Canvas) -> Canvas:
        """
        Render the given data onto the canvas within the bounding box.

        This method should modify the canvas in place and return it.

        Parameters:
        - data: Arbitrary data to render (e.g., text, points, shapes).
        - bbox: (x, y, width, height) bounding box where rendering should occur.
        - canvas: Numpy array representing the image to be modified.

        Returns:
        - The modified canvas (same object, modified in place).
        """
        pass


class BoxRenderer(Renderer):
    """
    Renders a bounding box inside a given bounding box region on a canvas.
    The rendered box is defined by normalized or absolute (x, y, w, h) coordinates.
    """

    def __init__(
        self,
        data_streamer: DataStreamer,
        color: Tuple[int, int, int] = (0, 255, 0),  # Default green
        thickness: int = 2
    ):
        """
        Parameters:
        - data_streamer: DataStreamer providing (x, y, w, h) for boxes.
        - color: BGR color of the box.
        - thickness: Border thickness.
        """
        super().__init__(data_streamer)
        self.default_color = color
        self.default_thickness = thickness

    def compute_size(self) -> OptionalSize:
        """Bounding box renderer does not occupy fixed space."""
        return None

    def render(self, data: Any, bbox: BoundingBox, canvas: Canvas) -> Canvas:
        """
        Renders a bounding box relative to the given bounding box area.

        Parameters:
        - data: A tuple (x, y, w, h), normalized (0–1) or absolute.
        - bbox: The parent bounding box (x0, y0, w0, h0) within which to render.
        - canvas: The image to draw on.

        Returns:
        - The modified canvas.
        """
        if not isinstance(data, (tuple, list)) or len(data) != 4:
            raise ValueError("Expected data to be a tuple of (x, y, width, height).")

        x0, y0, w0, h0 = bbox
        x, y, w, h = data

        # Determine if normalized (all values between 0 and 1)
        is_normalized = all(0.0 <= v <= 1.0 for v in (x, y, w, h))

        if is_normalized:
            x = int(x * w0)
            y = int(y * h0)
            w = int(w * w0)
            h = int(h * h0)
        else:
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

        # Offset by parent bbox
        x1 = x0 + x
        y1 = y0 + y
        x2 = x1 + w
        y2 = y1 + h

        # Draw the rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), self.default_color, self.default_thickness)
        return canvas


class COCOKeypoints3DRenderer(Renderer):
    """
    Renders 3D COCO-format keypoints inside a bounding box using matplotlib.
    """

    def __init__(
        self,
        data_streamer: DataStreamer,
        figsize: Tuple[int, int] = (4, 4),
        elev: int = 10,
        azim: int = -90,
        confidence_threshold: float = 0.0
    ):
        super().__init__(data_streamer)
        self.figsize = figsize
        self.elev = elev
        self.azim = azim
        self.confidence_threshold = confidence_threshold

    def compute_size(self) -> OptionalSize:
        return (None, None)

    def _render_3d_pose(self, pose: np.ndarray) -> np.ndarray:
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=self.elev, azim=self.azim)

        pose = np.asarray(pose)
        if pose.ndim != 2 or pose.shape[1] < 3:
            plt.close(fig)
            return None

        # Filter by confidence if present
        if pose.shape[1] == 4:
            conf = pose[:, 3]
        else:
            conf = np.ones(pose.shape[0])

        mask = conf >= self.confidence_threshold
        x, y, z = pose[mask, 0], pose[mask, 1], pose[mask, 2]

        ax.scatter(x, y, z, c='red', s=20)
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')

        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.get_size_inches() * fig.get_dpi()
        img = buf.reshape(int(h), int(w), 3)
        plt.close(fig)
        return img

    def render(self, data: Any, bbox: BoundingBox, canvas: Canvas) -> Canvas:
        x, y, w, h = bbox
        img = self._render_3d_pose(data)
        if img is None:
            return canvas

        img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        canvas[y:y+h, x:x+w] = img_resized
        return canvas


class COCOKeypointsRenderer(Renderer):
    """
    Renders COCO-format keypoints on a canvas within a bounding box.
    """

    def __init__(
        self,
        data_streamer: DataStreamer,
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
        - data_streamer: DataStreamer providing pose keypoints.
        - color: Circle color (BGR).
        - radius: Radius of each keypoint circle.
        - thickness: Thickness of the circle outline. -1 = filled.
        - draw_labels: Whether to draw labels on keypoints.
        - keypoint_labels: Mapping from index to string label (e.g., {4: 'LW'}).
        - font_scale: Scale of font used for labels.
        - font_color: Font color for labels.
        - font_thickness: Thickness of label text.
        - font_face: Font face for label text.
        - confidence_threshold: Minimum confidence to show a keypoint.
        - assume_normalized: If True/False, forces normalized/pixel input. If None, auto-detects.
        """
        super().__init__(data_streamer)
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
    
    def compute_size(self) -> OptionalSize:
        return (None, None)  # No fixed size, depends on the frame

    def _is_normalized(self, pose: np.ndarray) -> bool:
        # Heuristic: if any keypoint is > 2, it's probably pixel-based
        if self.assume_normalized is not None:
            return self.assume_normalized
        return np.max(pose[:, :2]) <= 1.0

    def _render_pose(self, pose: np.ndarray, canvas: Canvas, bbox: BoundingBox):
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

    def render(self, data: Any, bbox: BoundingBox, canvas: Canvas) -> Canvas:
        """
        Render COCO-style pose keypoints.

        Parameters:
        - data: One pose (np.ndarray of shape (K, 2 or 3)) or dict[int, pose]
        - bbox: Bounding box assumed to contain the full frame (x, y, w, h)
        - canvas: The canvas image to modify.

        Returns:
        - Modified canvas.
        """
        if isinstance(data, dict):
            for pose in data.values():
                self._render_pose(pose, canvas, bbox)
        else:
            self._render_pose(data, canvas, bbox)

        return canvas


class FrameRenderer(Renderer):
    """
    Renders an image frame (RGB or grayscale) onto a canvas within a bounding box.
    Optionally resizes the frame to the specified width and height, preserving aspect ratio.
    """
    def __init__(
        self,
        data_streamer: DecordVideoStreamer | StaticHorizontalLabelBarStreamer,
        interpolation: int = cv2.INTER_LINEAR,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """
        Parameters:
        - data_streamer: DataStreamer providing video frames.
        - interpolation: Interpolation method used for resizing.
        - width: Optional target width for rendering frames.
        - height: Optional target height for rendering frames.
        """
        super().__init__(data_streamer)
        self.interpolation = interpolation
        self.width = width
        self.height = height

    def compute_size(self) -> OptionalSize:
        return (self.width or self.data_streamer.width,
                self.height or self.data_streamer.height)

    def render(self, data: Any, bbox: BoundingBox, canvas: Canvas) -> Canvas:
        """
        Draws the given frame into the specified bounding box on the canvas.

        Parameters:
        - data: A numpy image (RGB or grayscale).
        - bbox: Bounding box (x, y, width, height) to draw into.
        - canvas: Image canvas to modify.

        Returns:
        - The modified canvas.
        """
        x, y, w, h = bbox

        if not isinstance(data, np.ndarray):
            raise TypeError("Expected data to be a numpy ndarray representing an image.")

        # Convert to BGR
        if len(data.shape) == 2 or data.shape[2] == 1:
            frame = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        elif data.shape[2] == 3:
            frame = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported image format. Expected 2D grayscale or 3D RGB/BGR image.")

        # Resize while preserving aspect ratio
        fh, fw = frame.shape[:2]
        target_w = self.width or fw
        target_h = self.height or fh

        scale = min(w / fw, h / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=self.interpolation)

        # Compute top-left corner for centering
        offset_x = x + (w - new_w) // 2
        offset_y = y + (h - new_h) // 2

        # Paste resized frame into canvas
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        return canvas


class ProgressRenderer(Renderer):
    """
    Renders a horizontal progress bar as a vertical rectangle sweeping
    from left to right over the canvas, based on a float progress value.
    """

    def __init__(self,
                 data_streamer: DataStreamer,
                 bar_color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2):
        """
        Initializes the progress renderer.

        Parameters:
        - bar_color: Color of the moving progress bar (BGR).
        - thickness: Width of the vertical progress bar rectangle.
        """
        super().__init__(data_streamer)
        self.bar_color = bar_color
        self.thickness = thickness

    def compute_size(self) -> OptionalSize:
        """
        Returns the fixed size of the progress bar canvas.
        """
        return (None, None)  # No fixed size, depends on the canvas

    def render(self, data: Any, bbox: BoundingBox, canvas: Canvas) -> Canvas:
        """
        Draws a vertical progress bar rectangle inside the given bounding box.

        Parameters:
        - data: A float between 0.0 and 1.0 indicating progress.
        - bbox: Bounding box (x, y, width, height) to draw within.
        - canvas: Full image canvas to draw on.

        Returns:
        - The modified canvas with progress drawn.
        """
        if not isinstance(data, float):
            raise TypeError("ProgressRenderer expects a float between 0.0 and 1.0.")
        if not (-0.05 <= data <= 1.05):  # Allow a little tolerance
            raise ValueError("Progress value must be between 0.0 and 1.0.")
        data = max(0.0, min(1.0, data))  # Clamp to [0.0, 1.0]

        x, y, w, h = bbox
        progress_x = int(w * data)

        # Compute vertical bar position within the bounding box
        half_thick = self.thickness // 2
        bar_x1 = x + max(0, progress_x - half_thick)
        bar_x2 = x + min(w - 1, progress_x + half_thick)
        bar_y1 = y
        bar_y2 = y + h - 1

        # Draw vertical progress bar within the bbox
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), self.bar_color, thickness=-1)

        return canvas


class TextRenderer(Renderer):
    """
    Renders text within a bounding box on an image canvas using OpenCV.
    """

    def __init__(
        self,
        data_streamer: DataStreamer,
        font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.5,
        font_color: Tuple[int, int, int] = (0, 0, 0),  # Black (BGR)
        thickness: int = 1,
        line_type: int = cv2.LINE_AA,
        num_expected_lines: int = 1,
        float_precision: Optional[int] = None
    ):
        """
        Initialize the text renderer with configurable appearance.

        Parameters:
        - data_streamer: DataStreamer providing text data.
        - font_face: OpenCV font (e.g., cv2.FONT_HERSHEY_SIMPLEX)
        - font_scale: Scale of the font (float)
        - font_color: Font color in BGR format (tuple of 3 ints)
        - thickness: Thickness of the text lines
        - line_type: OpenCV line type (e.g., cv2.LINE_AA)
        - num_expected_lines: Number of lines expected in the text. Used to estimate height.
        - float_precision: If set, round floats in text to this precision
        """
        super().__init__(data_streamer)
        self.font_face = font_face
        self.font_scale = font_scale
        self.font_color = font_color
        self.thickness = thickness
        self.line_type = line_type
        self.num_expected_lines = num_expected_lines
        self.float_precision = float_precision

    def compute_size(self) -> OptionalSize:
        """Just horizontal text for now."""
        # Estimate text size
        (_, text_h), _ = cv2.getTextSize(
            "test", self.font_face, self.font_scale, self.thickness
        )
        return (None, text_h * self.num_expected_lines)

    def render(self, data: Any, bbox: BoundingBox, canvas: Canvas) -> Canvas:
        """
        Render text data onto the canvas inside the given bounding box.

        Parameters:
        - data: A dictionary containing the key 'text' and optionally other text attributes.
        - bbox: (x, y, width, height) bounding box
        - canvas: Image array to modify

        Returns:
        - Modified canvas
        """
        x, y, w, h = bbox

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
        text = f"{text:.{self.float_precision}f}" if isinstance(text, float) and self.float_precision is not None else str(text)

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
            line_type
        )

        return canvas