from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
from data.visualization.utils import BoundingBox, Canvas, OptionalSize

import cv2
import numpy as np
from typing import Any, Tuple, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from data.visualization.data_streamer import SizedDataStreamer

class Renderer(ABC):
    """
    Abstract base class for rendering arbitrary data.
    
    The data is rendered onto a canvas within a specified bounding box.
    """

    @property
    @abstractmethod
    def expected_size(self) -> OptionalSize:
        """
        Abstract property: Subclasses must implement this to return the
        expected (width, height) in pixels that this renderer ideally targets
        for rendering its data effectively.

        This helps in layout calculations to allocate appropriate space.
        """
        pass

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
        color: Tuple[int, int, int] = (0, 255, 0),  # Default green
        thickness: int = 2
    ):
        """
        Parameters:
        - data_streamer: DataStreamer providing (x, y, w, h) for boxes.
        - color: BGR color of the box.
        - thickness: Border thickness.
        """
        super().__init__()
        self.default_color = color
        self.default_thickness = thickness

    @property
    def expected_size(self) -> OptionalSize:
        return (None, None)  # No fixed size, depends on the canvas

    def render(self, data: Any, bbox: BoundingBox, canvas: Canvas) -> Canvas:
        """
        Renders bounding boxes relative to the given bounding box area.
        Assumes the provided bounding box is generate.

        Parameters:
        - data: A list of dicts with keys: "box", "score", "id", "text_label".
                "box" is (x1, y1, x2, y2), normalized or absolute.
        - bbox: The parent bounding box (x1, y1, x2, y2).
        - canvas: The image to draw on.

        Returns:
        - The modified canvas.
        """
        if data is None:
            return canvas

        if not isinstance(data, list):
            raise ValueError("Expected data to be a list of dictionaries.")

        x01, y01, x02, y02 = bbox
        w0 = x02 - x01
        h0 = y02 - y01

        for entry in data:
            box = entry.get("box")
            if box is None:
                continue

            if not isinstance(box, (tuple, list)) or len(box) != 4:
                raise ValueError("Each 'box' must be a tuple of (x1, y1, x2, y2).")

            x1, y1, x2, y2 = box
            is_normalized = all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2))

            if not is_normalized:
                x1 = float(x1) / w0
                y1 = float(y1) / h0
                x2 = float(x2) / w0
                y2 = float(y2) / h0
            
            x1 = int(x01 + x1 * w0)
            y1 = int(y01 + y1 * h0)
            x2 = int(x01 + x2 * w0)
            y2 = int(y01 + y2 * h0)

            # Draw bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), self.default_color, self.default_thickness)

            # Prepare annotation text
            label_parts = []
            if "text_label" in entry:
                label_parts.append(f"{entry['text_label']} ")
            if "id" in entry:
                label_parts.append(f"id:{entry['id']}")
            if "score" in entry:
                label_parts.append(f"{entry['score']:.2f}")
            label = " ".join(label_parts)

            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                # text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_origin = (x1, y1 - 5 if y1 - 5 > 10 else y1 + 15)
                cv2.putText(canvas, label, text_origin, font, font_scale, self.default_color, thickness, lineType=cv2.LINE_AA)

        return canvas



class COCOKeypoints3DRenderer(Renderer):
    """
    Renders 3D COCO-format keypoints inside a bounding box using matplotlib.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (4, 4),
        elev: int = 10,
        azim: int = -90,
        confidence_threshold: float = 0.0
    ):
        super().__init__()
        self.figsize = figsize
        self.elev = elev
        self.azim = azim
        self.confidence_threshold = confidence_threshold

    @property
    def expected_size(self) -> OptionalSize:
        return (None, None)  # No fixed size, depends on the canvas

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
        if data is None:
            return canvas

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
        super().__init__()
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
    def expected_size(self) -> OptionalSize:
        return (None, None)  # No fixed size, depends on the canvas

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
        if data is None:
            return canvas

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
        interpolation: int = cv2.INTER_LINEAR,
        width: Optional[int] = None,
        height: Optional[int] = None,
        sized_streamer: Optional[SizedDataStreamer] = None,
    ):
        """
        Parameters:
        - data_streamer: DataStreamer providing video frames.
        - interpolation: Interpolation method used for resizing.
        - sized_streamer: Optional SizedDataStreamer to provide size hints.
        - width: Optional target width for rendering frames.
        - height: Optional target height for rendering frames.

        Width and height are given preference over the size deduced from the sized streamer.
        While the width, height, and sized streamer do not need to provide a known size,
        the size needs to be calculable from the layout.
        """
        super().__init__()
        self._interpolation = interpolation
        self._width = width
        self._height = height
        self._sized_streamer = sized_streamer

    @property
    def expected_size(self) -> OptionalSize:
        streamer_width, streamer_height = self._sized_streamer.size
        width = self._width if self._width is not None else streamer_width
        height = self._height if self._height is not None else streamer_height
        return (width, height)

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
        if data is None:
            return canvas

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
        target_w = self._width or fw
        target_h = self._height or fh

        scale = min(w / fw, h / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=self._interpolation)

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
                 bar_color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2):
        """
        Initializes the progress renderer.

        Parameters:
        - bar_color: Color of the moving progress bar (BGR).
        - thickness: Width of the vertical progress bar rectangle.
        """
        super().__init__()
        self.bar_color = bar_color
        self.thickness = thickness

    @property
    def expected_size(self) -> OptionalSize:
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
        if data is None:
            return canvas

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
    

class HorizontalLabelBarRenderer(Renderer):
    """Useful for timestep-dependent labels in visualizations."""
    def __init__(self,
                 height: int = 20,
                 color_seed: Optional[int] = None,
    ) -> None:
        """
        :param height: Height of the label bar in pixels.
        :param color_seed: Optional seed to keep label colors consistent.
        """
        super().__init__()
        self._height = height
        self._label_bar = None
        self._color_seed = color_seed if color_seed is not None else 42
        self._colors = None
    
    @property
    def expected_size(self) -> OptionalSize:
        return (None, self._height)

    def _create_label_bar(self, labels: List[str], bar_height: int, bar_width: int) -> Dict[str, tuple]:
        """Assign consistent BGR colors to label strings."""
        unique_labels = sorted(set(labels))
        n = len(unique_labels)
        if n > 10:
            raise ValueError("Too many unique labels for a bar plot (must be ≤ 10).")
        
        cmap = plt.get_cmap("tab10")
        colors = [tuple(int(255 * c) for c in to_rgb(cmap(i))) for i in range(n)]
        colors = [tuple(reversed(color)) for color in colors]  # Convert RGB to BGR
        self._colors = {label: colors[i] for i, label in enumerate(unique_labels)}
        self._label_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

        total_samples = len(labels)
        segment_width = float(bar_width) / total_samples

        for i, label in enumerate(labels):
            color = self._colors[label]
            start = int(i * segment_width)
            end = int((i + 1) * segment_width)
            self._label_bar[:, start:end] = color

    def render(self, data: List[str], bbox: BoundingBox, canvas: Canvas) -> Canvas:
        """
        Draw a horizontal label bar within the bounding box on the canvas.
        Each label's proportion is shown using a unique color.

        :param data: List of label strings.
        :param bbox: Bounding box (x1, y1, x2, y2) within which to draw the label bar.
        :param canvas: The image canvas (numpy array) to draw on.
        :return: The modified canvas.
        """
        if data is None:
            return canvas

        x_offset, y_offset, bar_width, bar_height = bbox

        if bar_width <= 0 or bar_height <= 0:
            return canvas  # Nothing to draw

        if self._label_bar is None:
            self._create_label_bar(data, bar_height, bar_width)

        canvas[y_offset:y_offset+bar_height, x_offset:x_offset+bar_width] = self._label_bar
        return canvas


class TextRenderer(Renderer):
    """
    Renders text within a bounding box on an image canvas using OpenCV.
    """
    def __init__(
        self,
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
        super().__init__()
        self.font_face = font_face
        self.font_scale = font_scale
        self.font_color = font_color
        self.thickness = thickness
        self.line_type = line_type
        self.num_expected_lines = num_expected_lines
        self.float_precision = float_precision

    @property
    def expected_size(self) -> OptionalSize:
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

        text = f"{text:.{self.float_precision}f}" if isinstance(text, float) and self.float_precision is not None else str(text)

        x, y, w, h = bbox
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
