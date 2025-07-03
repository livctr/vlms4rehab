from typing import Any, Tuple, Dict
import cv2
import numpy as np

from vidplot.core import Renderer, DataStreamer


def paint_box_in_place(
    image_array: np.ndarray,
    bbox_coords: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    label: str = None,
    thickness: int = 2,
    font_scale: float = 0.5,
):
    """
    Draws a bounding box and an optional label on an image in place.

    Args:
        image_array: The NumPy array of the image to draw on.
        bbox_coords: A tuple (x1, y1, x2, y2) for the box coordinates.
        color: The RGB color for the box and label background.
        label: The text to display on the label.
        thickness: The thickness of the box lines.
        font_scale: The scale of the label font.
    """
    x1, y1, x2, y2 = bbox_coords

    # Draw the main bounding box rectangle
    cv2.rectangle(image_array, (x1, y1), (x2, y2), color, thickness)

    if label:
        # Set up font and calculate the size of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Create a filled rectangle for the label's background
        label_bg_y2 = y1
        label_bg_y1 = y1 - text_height - baseline

        # Ensure the label does not go off the top of the screen
        label_bg_y1 = max(label_bg_y1, 0)

        cv2.rectangle(image_array, (x1, label_bg_y1), (x1 + text_width, label_bg_y2), color, -1)

        # Put the label text on top of the background
        # Use white text for better contrast against a colored background
        text_y = y1 - baseline // 2
        cv2.putText(image_array, label, (x1, text_y), font, font_scale, (255, 255, 255), 1)


class BoxRenderer(Renderer):
    """
    A renderer for overlaying bounding boxes on a video frame.

    This renderer draws bounding boxes for objects, with optional labels
    displaying ID and score. It correctly handles resizing to align with a
    base video layer.
    """

    def __init__(
        self,
        name: str,
        data_streamer: DataStreamer,
        grid_row: Tuple[int, int],
        grid_column: Tuple[int, int],
        id_to_color: Dict[int, Tuple[int, int, int]],
        box_representation_format: str = "xyxy",
        label_box: bool = True,
        line_thickness: int = 2,
        font_scale: float = 0.5,
        resize_mode: str = "fit",
        z_index: int = 1,
    ):
        super().__init__(name, data_streamer, grid_row, grid_column, z_index=z_index)
        assert resize_mode in ("fit", "stretch", "center")
        assert box_representation_format in ("xyxy", "xywh")

        self.id_to_color = id_to_color
        self.box_representation_format = box_representation_format
        self.label_box = label_box
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.resize_mode = resize_mode

    def _default_size(self) -> Tuple[int, int]:
        return (100, 100)

    def _render(self, data: Any, bbox: Tuple[int, int, int, int], canvas: Any) -> Any:
        """
        Renders bounding boxes on top of the existing canvas content.

        This method processes a list of box detections, transforms their
        coordinates based on the resize mode, and paints them onto the canvas.
        The expected data format is a dictionary:
        {
            "shape": (original_height, original_width),
            "boxes": [
                {"box": [x1, y1, x2, y2], "score": 0.9, "id": 1},
                ...
            ]
        }
        """
        canvas_x1, canvas_y1, canvas_x2, canvas_y2 = bbox
        target_w, target_h = canvas_x2 - canvas_x1, canvas_y2 - canvas_y1

        if data is None or not data.get("boxes"):
            return canvas

        # Store shape if not already stored, or use stored shape
        if not hasattr(self, "_cached_shape"):
            self._cached_shape = data["shape"]
        fh, fw = self._cached_shape

        for item in data["boxes"]:
            box_coords = item["box"]
            box_id = item.get("id")

            if box_id is None or box_id not in self.id_to_color:
                continue

            color = self.id_to_color[box_id]

            # 1. Convert box to xyxy format
            if self.box_representation_format == "xywh":
                x1, y1, w, h = box_coords
                x2, y2 = x1 + w, y1 + h
            else:
                x1, y1, x2, y2 = box_coords

            # 2. Apply resize transformation to get coordinates relative to the target view
            final_box = None
            if self.resize_mode == "stretch":
                x_scale, y_scale = target_w / fw, target_h / fh
                nx1, ny1 = int(x1 * x_scale), int(y1 * y_scale)
                nx2, ny2 = int(x2 * x_scale), int(y2 * y_scale)
                final_box = (nx1, ny1, nx2, ny2)

            elif self.resize_mode == "fit":
                scale = min(target_w / fw, target_h / fh)
                new_w, new_h = int(fw * scale), int(fh * scale)
                x_off, y_off = (target_w - new_w) // 2, (target_h - new_h) // 2
                nx1, ny1 = int(x1 * scale + x_off), int(y1 * scale + y_off)
                nx2, ny2 = int(x2 * scale + x_off), int(y2 * scale + y_off)
                final_box = (nx1, ny1, nx2, ny2)

            elif self.resize_mode == "center":
                x_off, y_off = (target_w - fw) // 2, (target_h - fh) // 2
                nx1, ny1 = int(x1 + x_off), int(y1 + y_off)
                nx2, ny2 = int(x2 + x_off), int(y2 + y_off)
                # Clip the box to the target view's boundaries
                nx1_c = max(0, nx1)
                ny1_c = max(0, ny1)
                nx2_c = min(target_w, nx2)
                ny2_c = min(target_h, ny2)
                if nx1_c < nx2_c and ny1_c < ny2_c:
                    final_box = (nx1_c, ny1_c, nx2_c, ny2_c)

            # 3. Draw the final box on the canvas
            if final_box:
                # Convert to absolute canvas coordinates
                abs_x1 = final_box[0] + canvas_x1
                abs_y1 = final_box[1] + canvas_y1
                abs_x2 = final_box[2] + canvas_x1
                abs_y2 = final_box[3] + canvas_y1

                label = None
                if self.label_box:
                    score = item.get("score")
                    label = f"ID: {box_id}"
                    if score is not None:
                        label += f" ({score:.2f})"

                paint_box_in_place(
                    canvas,
                    (abs_x1, abs_y1, abs_x2, abs_y2),
                    color=color,
                    label=label,
                    thickness=self.line_thickness,
                    font_scale=self.font_scale,
                )
        return canvas
