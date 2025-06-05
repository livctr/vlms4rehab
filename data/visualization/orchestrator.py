import os
import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from collections import deque
from tqdm import tqdm
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple, Union

from data.visualization.data_streamer import DataStreamer
from data.visualization.renderer import Renderer
from data.visualization.utils import (
    Layout,
    OptionalSize,
    RelativePosition,
    RoughSpatialPosition,
    Size, 
    BoundingBox,
    logger
)


class Orchestrator(ABC):
    """
    A node in a rendering tree, responsible for spatial layout and orchestrating
    rendering. See `CompositeOrchestrator` and `LeafOrchestrator`
    for concrete implementations. The leaf orchestrator holds the streaming data,
    while the composite orchestrator manages their layout.
    """
    def __init__(self) -> None:
        self._fps: Optional[float] = None

    @property
    def fps(self) -> float:
        """
        Returns the frames per second (FPS) of the orchestrator.
        This is used to determine the sample rate for data streamers.
        """
        return self._fps

    def compile(self, sample_rate: Union[float, int]) -> None:
        """
        Compiles this node and its children into a layout and synchronizes the data streamers.

        Subclasses should implement this so that `get_layout` returns a meaningful values.
        """
        self._fps = 1. / sample_rate
    
    def get_approx_length(self) -> Union[int, float]:
        """
        Returns the approximate length of the data stream.
        """
        return float('inf')

    @abstractmethod
    def get_layout(self) -> Layout:
        """
        Returns the layout of the node, which is a list of tuples (id, bounding_box),
        where id corresponds to the streaming data.
        """
        pass

    @abstractmethod
    def get_renderers(self) -> Dict[str, Renderer]:
        """
        Returns a dictionary mapping node IDs to their corresponding renderers orchestrated
        by this node.
        """
        pass

    @abstractmethod
    def get_size(self) -> OptionalSize:
        """
        Returns the input size of the node as a tuple (width, height). Possible that one or both
        of the dimensions are None. Used to construct the layout of the node.
        """
        pass

    @abstractmethod
    def next(self) -> Dict[str, Tuple[float, Any]]:
        """
        Returns the next data item for each child node in the layout.
        This method is called to retrieve the next data item for rendering.

        Returns:
            Dict[str, Tuple[float, Any]]: A dictionary mapping 
        """
        pass


class CompositeOrchestrator(Orchestrator):

    def __init__(self, gap: int = 5) -> None:
        """
        Initializes a composite orchestrator with an optional gap between children.

        Args:
            gap (int): The gap in pixels between children nodes in the layout. Defaults to 5.
        """
        super().__init__()
        self.gap = gap
        self._children: List[Orchestrator] = []  # Children nodes, sorted by layer ascending
        self._raw_rough_positions: List[RoughSpatialPosition] = []
        self._rough_positions: List[RoughSpatialPosition] = []
        self._relative_positions: List[RelativePosition] = []

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
            child_size = child.get_size()
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

        self._size = (cum_x[-1], cum_y[-1])

        # Put in the relative positions of children
        self._relative_positions = []
        for i in range(len(self._children)):
            layer, row, col = self._rough_positions[i]
            self._relative_positions.append(
                (cum_x[col], cum_y[row], cum_x[col + 1] - cum_x[col], cum_y[row + 1] - cum_y[row], layer)
            )
        
        # Set the layout
        layout = []
        for i, child in enumerate(self._children):
            child_layout = child.get_layout()
            rel_x = self._relative_positions[i][0]
            rel_y = self._relative_positions[i][1]
            w = self._relative_positions[i][2]
            h = self._relative_positions[i][3]
            layout.extend([(id_, (x + rel_x, y + rel_y, w, h)) for id_, (x, y, _, _) in child_layout])
        self._layout = layout

    def get_approx_length(self) -> Union[int, float]:
        """Approximate length of the data stream."""
        return min([child.get_approx_length() for child in self._children], default=float('inf'))

    def get_layout(self) -> Layout:
        return self._layout
    
    def get_renderers(self) -> Dict[str, Renderer]:
        renderers = {}
        for child in self._children:
            child_renderers = child.get_renderers()
            renderers.update(child_renderers)
        return renderers

    def get_size(self) -> OptionalSize:
        return self._size

    def next(self) -> Dict[str, Tuple[float, Any]]:
        data = {}
        for child in self._children:
            child_data = child.next()
            data.update(child_data)
        return data

    def add_children(self, children: List[Orchestrator], rough_positions: List[RoughSpatialPosition]) -> None:
        """
        Adds multiple children with their rough spatial positions.

        The children and their positions are sorted by layer (ascending).
        Rough positions (layer, row, column) are then compacted into zero-based indices.

        Args:
            children: List of Orchestrator instances to add as children.
            rough_positions: List of tuples (layer, row, column) for each child.
                             These define the child's intended position in a conceptual grid layout.

        Raises:
            ValueError: If the number of children does not match the number of rough positions,
                        or if children are added to a node that has an `initial_size` set
                        (as such nodes are expected to be sized explicitly, not by content).
        """
        if len(children) != len(rough_positions):
            raise ValueError("Number of children must match number of rough positions.")
        
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


class LeafOrchestrator(Orchestrator):

    def __init__(self, id: str, streamer: DataStreamer, renderer: Renderer, initial_size: OptionalSize = (None, None)):
        super().__init__()
        self._initial_size = initial_size
        self.id = id
        self._streamer = streamer
        self._renderer = renderer

    def compile(self, sample_rate) -> None:
        super().compile(sample_rate)
        self._streamer.sample_rate = sample_rate
        if self._initial_size[0] is None and self._initial_size[1] is None:
            self._size = self._renderer.expected_size
        else:
            self._size = self._initial_size
        logger.info(f"Leaf {self.id} has input size: {self._size} and sample rate (seconds per sample): {sample_rate:.2f} s")
    
    def get_approx_length(self) -> Union[int, float]:
        """Approximate length of the data stream."""
        return self._streamer.approx_length

    def get_size(self) -> OptionalSize:
        return self._size

    def get_layout(self) -> Layout:
        """
        Returns the local layout of the leaf node, which is simply its initial size.
        """
        return [(self.id, (0, 0, self._size[0], self._size[1]))]

    def get_renderers(self) -> Dict[str, Renderer]:
        return {self.id: self._renderer}

    def next(self) -> Dict[str, Tuple[float, Any]]:
        return {self.id: next(self._streamer)}


def _write_to_file_single_process(fourcc_str: str,
                                  file_path: str,
                                  fps: float,
                                  size: Size,
                                  layout: List[Tuple[str, BoundingBox]],
                                  renderers: Dict[str, Renderer],
                                  node: Orchestrator,
                                  approx_length: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out = cv2.VideoWriter(file_path, fourcc, fps, (size[0], size[1]))
    progress_bar_total = approx_length if approx_length > 0 else None
    progress_bar = tqdm(desc=f"Processing video {file_path} (single-thread)", total=progress_bar_total, unit="frame", smoothing=0.1)

    actual_frames_written = 0

    while True:

        canvas_height, canvas_width = size[1], size[0]
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        try:
            data = node.next()
            for renderer_id, bbox in layout:
                _, streamer_data = data[renderer_id]
                renderers[renderer_id].render(streamer_data, bbox, canvas)
            out.write(canvas)
            actual_frames_written += 1
            progress_bar.update(1)
        except StopIteration:
            logger.debug("Reached end of data stream.")
            break
        except Exception as e:
            logger.error(f"Error during rendering: {e}", exc_info=True)
            raise e

    progress_bar.close()
    out.release()
    logger.info(f"Successfully wrote video to {file_path} with {actual_frames_written} frames.")


def _render_frame_for_mp(task_payload: Tuple[Any, List[Tuple[str, BoundingBox]], Dict[str, Renderer], Size]) -> np.ndarray:
    """
    Worker function to render a single frame. Executed in a separate process.
    
    Args:
        task_payload: A tuple containing (node_data, layout, renderers_dict, canvas_size_tuple).
                      - node_data: Data for the frame, obtained from node.next().
                      - layout: Layout information for rendering.
                      - renderers_dict: Dictionary of renderer objects.
                      - canvas_size_tuple: (width, height) of the canvas.
    
    Returns:
        A NumPy array representing the rendered canvas (frame).
    """
    node_data, layout, renderers_dict, canvas_size_tuple = task_payload

    canvas_width, canvas_height = canvas_size_tuple[0], canvas_size_tuple[1]
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    for renderer_id, bbox in layout:
        _, streamer_data = node_data[renderer_id]
        renderers_dict[renderer_id].render(streamer_data, bbox, canvas)

    return canvas


def _write_to_file_multiprocess(fourcc_str: str,
                                file_path: str,
                                fps: float,
                                size: Size,
                                layout: List[Tuple[str, BoundingBox]],
                                renderers: Dict[str, Renderer],
                                node: Orchestrator,
                                approx_length: int,
                                num_workers: int = None) -> None:
    """
    Writes video to a file using multiple processes for rendering frames.

    Data fetching (`node.next()`) occurs in the main process. Frame rendering
    is offloaded to a pool of worker processes. Rendered frames are written
    to the video file sequentially in the main process to maintain order.

    Args:
        fourcc_str: String for FourCC code (e.g., "MJPG", "XVID").
        file_path: Path to save the output video file.
        fps: Frames per second for the output video.
        size: Tuple (width, height) for the video frames.
        layout: A list of tuples, where each tuple contains a renderer ID (str)
                and its BoundingBox.
        renderers: A dictionary mapping renderer IDs to Renderer objects.
                   These objects (and their components) must be picklable.
        node: An Orchestrator object with a `next()` method that yields data
              for each frame.
        approx_length: Approximate number of frames for the progress bar.
                       If <= 0, the progress bar won't show a total.
        num_workers: Number of worker processes for rendering. If None,
                     defaults to the number of CPUs on the system.
    """
    if not (isinstance(size, tuple) and len(size) == 2 and
            isinstance(size[0], int) and isinstance(size[1], int)):
        logger.error("Argument 'size' must be a tuple of two integers (width, height).")
        raise ValueError("Argument 'size' must be a tuple of two integers (width, height).")

    video_writer_width, video_writer_height = size[0], size[1]
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out = cv2.VideoWriter(file_path, fourcc, fps, (video_writer_width, video_writer_height))
    
    actual_num_workers = num_workers
    if actual_num_workers is None:
        actual_num_workers = os.cpu_count()
        if actual_num_workers is None:  # Fallback if os.cpu_count() is None
            actual_num_workers = 2
            logger.warning("Could not determine CPU count, defaulting to 2 workers.")
    elif actual_num_workers <= 0:
        logger.warning(f"num_workers was {actual_num_workers}, defaulting to 1 worker.")
        actual_num_workers = 1

    progress_bar_total = approx_length if approx_length > 0 else None
    progress_bar_desc = f"Processing video {file_path} (multi-process, {actual_num_workers} workers)"
    progress_bar = tqdm(desc=progress_bar_desc, total=progress_bar_total, unit="frame", smoothing=0.1)

    actual_frames_written = 0
    
    # max_inflight_tasks: Limits how many frames are read ahead and submitted for rendering.
    # This helps manage memory and keeps a steady flow of tasks to workers.
    max_inflight_tasks = actual_num_workers * 2
    if max_inflight_tasks <= 0: # Should only happen if actual_num_workers became <=0
         max_inflight_tasks = 1

    submitted_futures = deque()  # Stores futures of submitted rendering tasks

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual_num_workers) as executor:
            node_exhausted = False  # True if node.next() raised StopIteration

            while True:
                # --- 1. Submit new rendering tasks ---
                # Submit if the node has more data and we're below the inflight task limit.
                while not node_exhausted and len(submitted_futures) < max_inflight_tasks:
                    try:
                        node_data = node.next()  # Fetch data for the next frame
                        task_payload = (node_data, layout, renderers, size)
                        future = executor.submit(_render_frame_for_mp, task_payload)
                        submitted_futures.append(future)
                    except StopIteration:
                        logger.debug("Reached end of data stream from node (StopIteration).")
                        node_exhausted = True
                        break  # Exit submission loop
                    except Exception as e:
                        logger.error(f"Error fetching data from node: {e}", exc_info=True)
                        for f_cancel in submitted_futures: # Cancel pending tasks
                            f_cancel.cancel()
                        raise # Propagate error

                # --- 2. Process rendered frames (oldest first) ---
                if not submitted_futures:
                    if node_exhausted:
                        # Node is exhausted, and all submitted tasks have been processed. Done.
                        break  # Exit the main while loop
                    else:
                        # Node not exhausted, but no futures currently.
                        # This can happen if node.next() is slow or at the very start.
                        # The outer loop will continue and attempt to submit more tasks.
                        # If node.next() previously errored, it would have been re-raised.
                        pass # Continue to allow more submissions or wait for node data
                
                if submitted_futures: # If there are tasks to process
                    oldest_future = submitted_futures[0] # Peek at the oldest task
                    try:
                        # Wait for the oldest task to complete (blocks).
                        canvas = oldest_future.result()
                        submitted_futures.popleft() # Remove from deque after success

                        out.write(canvas)
                        actual_frames_written += 1
                        progress_bar.update(1)
                    except concurrent.futures.CancelledError:
                        logger.warning("A rendering task was cancelled.")
                        if submitted_futures and submitted_futures[0] == oldest_future:
                            submitted_futures.popleft() # Ensure it's removed
                    except Exception as e:  # Catches errors from worker process execution
                        logger.error(f"Error during rendering a frame in a worker process: {e}", exc_info=True)
                        for f_cancel in submitted_futures: # Cancel all other pending tasks
                            f_cancel.cancel()
                        raise # Propagate the error from the worker
        
    except Exception as e:
        logger.error(f"Multiprocessing video writing failed critically: {e}", exc_info=True)
        raise # Re-raise to signal failure
    finally:
        if 'progress_bar' in locals() and progress_bar is not None:
            progress_bar.close()
        if 'out' in locals() and out.isOpened():
            out.release()
        logger.info(f"Video writing process finished for {file_path}. Total frames written: {actual_frames_written}.")


def write_to_file(node: Orchestrator,
                  file_path: str,
                  num_render_workers: int = -1,
                  use_multiprocessing: bool = True,
                  fourcc_str: str = 'mp4v'
) -> None:
    """
    Writes the layout and annotations of the Orchestrator to a video file.

    :param node: The *compiled* Orchestrator instance containing the layout and renderers.
    :param file_path: The path where the video file will be saved.
    :param num_render_workers: Number of worker processes for rendering (if use_multiprocessing is True).
                                Defaults to cpu_count() - 1, or 1 if cpu_count() is 1.
    :param use_multiprocessing: If True, uses multiprocessing for rendering. Otherwise, runs on the main thread.
    """
    # For logging
    layout = node.get_layout()
    approx_length = node.get_approx_length()
    fps = node.fps
    size = node.get_size()  # Assuming this returns (width, height) or (None, None)
    renderers = node.get_renderers()

    # Print a visually distinct header for layout information
    header = "Layout Configuration"
    separator = "=" * max(50, len(header))
    logger.info(separator)
    logger.info(header.center(len(separator)))
    logger.info(separator)

    # Print layout details in a structured format
    for id_, bbox in layout:
        x, y, w, h = bbox
        logger.info(f"• Renderer: {id_:<15}\tPosition (x,y): ({x:>4}, {y:>4})  Size (w,h): {w:>4} x {h:<4}")

    # Print video configuration
    logger.info(separator)
    logger.info("Video Configuration".center(len(separator)))
    logger.info(separator)
    logger.info(f"• Duration: {approx_length if approx_length > 0 else 'Unknown'} frames")
    logger.info(f"• Frame Rate: {fps} FPS")
    logger.info(f"• Number of renderers: {len(renderers)}")
    logger.info(separator)

    if use_multiprocessing:
        raise NotImplementedError("Multiprocessing rendering is not yet implemented.")
    else:
        _write_to_file_single_process(
            fourcc_str=fourcc_str,
            file_path=file_path,
            fps=fps,
            size=size,
            layout=layout,
            renderers=renderers,
            node=node,
            approx_length=approx_length
        )
