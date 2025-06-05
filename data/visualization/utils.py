from typing import Tuple, Any, List, Optional
import numpy as np


RoughSpatialPosition = Tuple[int, int, int]  # (layer, row, column)
BoundingBox = Tuple[int, int, int, int]  # (x, y, width, height)
RelativePosition = Tuple[int, int, int, int, int]  # (x, y, width, height, layer)
Canvas = np.ndarray
Layout = List[Tuple[str, BoundingBox]]
OptionalSize = Tuple[Optional[int], Optional[int]]  # (width, height)