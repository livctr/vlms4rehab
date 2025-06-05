from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
from data.visualization.renderer import Renderer
from data.visualization.utils import Layout, RoughSpatialPosition, RelativePosition, OptionalSize
from abc import ABC, abstractmethod


class RenderNode(ABC):
    """
    A node in a rendering tree, responsible for spatial layout and orchestrating
    the rendering of itself and its children.

    The node can either have a fixed initial size (for leaf nodes with inherent content)
    or calculate its size based on its children's layout. Leaf nodes are expected to
    """
    def __init__(self, initial_size: OptionalSize = (None, None)) -> None:
        self.renderers: Dict[str, Renderer] = {}  # Maps IDs to Renderers
        self.size = initial_size
        self.sample_rate: Optional[Union[float, int]] = None

    @property
    @abstractmethod
    def is_leaf(self) -> bool:
        """Returns True if the node has no children, False otherwise."""
        pass

    def compile(self, sample_rate: Union[float, int]) -> None:
        """
        Compiles the node and its children into a layout and sets the sampling rate for data streamers.
        This method should be overridden by subclasses to implement specific layout logic.
        """
        self.fps = 1. / sample_rate

    @abstractmethod
    def get_layout(self) -> Layout:
        """
        Returns the absolute layout of the streamers relative to this node.
        This method is called after the node has been compiled.

        Returns:
            Layout: The layout of the node.
        """
        pass

    @abstractmethod
    def next(self) -> Dict[str, Tuple[float, Any]]:
        """
        Returns the next data item for each child node in the layout.
        This method is called to retrieve the next data item for rendering.

        Returns:
            Dict[str, Tuple[float, Any]]: A dictionary mapping child IDs to their next data items.
        """
        pass


class CompositeNode(RenderNode):

    def __init__(self, gap: int = 5) -> None:
        super().__init__()
        self.gap = gap
        self._children: List[RenderNode] = []  # Children nodes, sorted by layer ascending
        self._raw_rough_positions: List[RoughSpatialPosition] = []
        self._rough_positions: List[RoughSpatialPosition] = []
        self._relative_positions: List[RelativePosition] = []

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    def compile(self, sample_rate: Union[float, int]) -> None:
        """
        Compiles by setting the size of this node and the relative positions of its children.

        - If the node is a leaf:
            - Uses the provided size.
            - Otherwise, it defaults to (0,0).
        - If the node has children:
            - Its size becomes the bounding box encompassing all children,
              arranged according to their compacted rough positions (layer, row, col).
            - It populates `self._relative_positions` for children. The width and height
              in `_relative_positions` correspond to the full dimensions of the grid
              cell allocated to the child.

        Returns:
            Tuple[int, int]: The calculated (height, width) of the node.

        Raises:
            ValueError: If the node has children but no corresponding rough positions,
                        or if rough positions are empty when children exist.
        """
        super().compile(sample_rate)
        nrows = 1 + max(pos[1] for pos in self._rough_positions)
        ncols = 1 + max(pos[2] for pos in self._rough_positions)

        # Store maximize size at each coordinate
        size_matrix = [[(0, 0) for _ in range(ncols)] for _ in range(nrows)]  # stored as matrix of height, width
        for i, child in enumerate(self._children):
            child = self._children[i]
            child.compile(sample_rate)
            child_size = child.size
            layer, row, col = self._rough_positions[i]

            # Get maximum size at each coordinate
            size_matrix[row][col] = (
                max(size_matrix[row][col][0], child_size[0]) if child_size[0] else size_matrix[row][col][0],
                max(size_matrix[row][col][1], child_size[1]) if child_size[1] else size_matrix[row][col][1]
            )

        cum_x = [0]
        for col in range(ncols):
            max_width = max(size_matrix[row][col][0]for row in range(nrows))
            if max_width == 0:
                raise ValueError(
                    f"None of the streamers in column {col} have a positive width specified. Cannot compile."
                )
            if col < ncols - 1:
                cum_x.append(cum_x[-1] + max_width + self.gap)
            else:
                cum_x.append(cum_x[-1] + max_width)
        cum_y = [0]
        for row in range(nrows):
            max_height = max(size_matrix[row][col][1] for col in range(ncols))
            if max_height == 0:
                raise ValueError(
                    f"None of the streamers in row {row} have a positive height specified. Cannot compile."
                )
            if row < nrows - 1:
                cum_y.append(cum_y[-1] + max_height + self.gap)
            else:
                cum_y.append(cum_y[-1] + max_height)

        self.size = (cum_x[-1], cum_y[-1])

        # Put in the relative positions of children
        self._relative_positions = []
        for i in range(len(self._children)):
            layer, row, col = self._rough_positions[i]
            self._relative_positions.append(
                (cum_x[col], cum_y[row], cum_x[col + 1] - cum_x[col], cum_y[row + 1] - cum_y[row], layer)
            )

    def get_layout(self) -> Layout:
        if self.size[0] is None or self.size[1] is None:
            raise ValueError("Cannot get layout of a composite node without a size. Make sure to call `compile()`.")

        layout = []
        for i, child in enumerate(self._children):
            child_layout = child.get_layout()
            rel_x = self._relative_positions[i][0]
            rel_y = self._relative_positions[i][1]
            w = self._relative_positions[i][2]
            h = self._relative_positions[i][3]
            layout.extend([(id_, (x + rel_x, y + rel_y, w, h)) for id_, (x, y, _, _) in child_layout])
        return layout

    def add_children(self, children: List[RenderNode], rough_positions: List[RoughSpatialPosition]) -> None:
        """
        Adds multiple children with their rough spatial positions.

        The children and their positions are sorted by layer (ascending).
        Rough positions (layer, row, column) are then compacted into zero-based indices.

        Args:
            children: List of RenderNode instances to add as children.
            rough_positions: List of tuples (layer, row, column) for each child.
                             These define the child's intended position in a conceptual grid layout.

        Raises:
            ValueError: If the number of children does not match the number of rough positions,
                        or if children are added to a node that has an `initial_size` set
                        (as such nodes are expected to be sized explicitly, not by content).
        """
        if len(children) != len(rough_positions):
            raise ValueError("Number of children must match number of rough positions.")
        
        for child in children:
            self.renderers.update(child.renderers)  # track all child renderers
        self._children.extend(children)
        self._raw_rough_positions.extend(rough_positions)
        combined = sorted(zip(self._children, self._raw_rough_positions),
                          key=lambda x: x[1][0])
        self._children = [item[0] for item in combined]
        self._raw_rough_positions = [item[1] for item in combined]
        # Compact the layers to be integers from [0, n-1] based on the provided order in 
        # rough positions. Do the same for rows and columns.
        comp_maps = []
        for j in range(3):
            comp = list(set(pos[j] for pos in self._raw_rough_positions))
            comp.sort()
            comp_map = {val: idx for idx, val in enumerate(comp)}
            comp_maps.append(comp_map)

        # Update rough positions to use indices of rows, columns, and layers
        self._rough_positions = [
            (comp_maps[0][pos[0]], comp_maps[1][pos[1]], comp_maps[2][pos[2]])
            for pos in self._raw_rough_positions
        ]

    def clear_children(self) -> None:
        """
        Clears all children and their rough positions.
        Resets the node to a leaf state.
        """
        self._children.clear()
        self._raw_rough_positions.clear()
        self._rough_positions.clear()
        self._relative_positions.clear()
    
    def next(self) -> Dict[str, Tuple[float, Any]]:
        data = {}
        for child in self._children:
            child_data = child.next()
            data.update(child_data)
        return data


class LeafNode(RenderNode):

    def __init__(self, id: str, renderer: Renderer, initial_size: OptionalSize = (None, None)):
        super().__init__(initial_size)
        self.id = id
        self._renderer = renderer
        self.renderers = {self.id: self._renderer}

    @property
    def is_leaf(self) -> bool:
        return True

    def compile(self, sample_rate) -> None:
        super().compile(sample_rate)
        self.size = self._renderer.compute_size()
        self._renderer.data_streamer.sample_rate = sample_rate

    def get_layout(self) -> Layout:
        """
        Returns the local layout of the leaf node, which is simply its initial size.
        """
        return [(self.id, (0, 0, self.size[0], self.size[1]))]

    def next(self) -> Dict[str, Tuple[float, Any]]:
        return {self.id: next(self._renderer.data_streamer)}
