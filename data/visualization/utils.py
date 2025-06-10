import logging
from typing import Tuple, List, Optional
import numpy as np


RoughSpatialPosition = Tuple[int, int, int]  # (layer, row, column)
BoundingBox = Tuple[int, int, int, int]  # (x, y, width, height)
RelativePosition = Tuple[int, int, int, int, int]  # (x, y, width, height, layer)
Canvas = np.ndarray
Layout = List[Tuple[str, BoundingBox]]
OptionalSize = Tuple[Optional[int], Optional[int]]  # (width, height)
Size = Tuple[int, int]  # (width, height)

logger = logging.getLogger("video_visualization")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers if imported multiple times
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
