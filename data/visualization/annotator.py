import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
import multiprocessing.queues # For type hinting, though might not be strictly necessary if only used in `except`
import logging

from data.visualization.layout import RenderNode # Assuming this import is correct

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _render_frame_worker(data_queue: multiprocessing.Queue,
                         frame_queue: multiprocessing.Queue,
                         layout: list,
                         renderers: dict,
                         canvas_size: tuple,
                         worker_id: int):
    """
    Worker process function to render frames.
    Retrieves data from data_queue, renders it, and puts the canvas into frame_queue.
    """
    logging.debug(f"Render worker {worker_id} started.")
    while True:
        item = data_queue.get()
        if item is None:
            logging.debug(f"Render worker {worker_id} received sentinel, putting sentinel to frame_queue.")
            frame_queue.put(None) # Signal writer that this worker is done
            break

        idx, data_dict = item
        canvas_height, canvas_width = canvas_size[1], canvas_size[0]
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        if data_dict is None:
            logging.debug(f"Render worker {worker_id} received None data_dict for index {idx}.")
            frame_queue.put((idx, canvas))
            continue

        try:
            for renderer_id, bbox in layout:
                if renderer_id in renderers:
                    if renderer_id in data_dict:
                        if data_dict[renderer_id] is not None:
                             # Assuming data_dict[renderer_id] is structured as (metadata, actual_data)
                             _, data_for_renderer = data_dict[renderer_id]
                             renderers[renderer_id].render(data_for_renderer, bbox, canvas)
                        else:
                            logging.debug(f"Worker {worker_id}: Data for renderer {renderer_id} is None at frame {idx}")
                    else:
                        logging.debug(f"Worker {worker_id}: No data key for renderer {renderer_id} in data_dict for frame {idx}")
                else:
                    logging.debug(f"Worker {worker_id}: Missing renderer_id {renderer_id} in renderers for frame {idx}")
            frame_queue.put((idx, canvas))
        except Exception as e:
            logging.warning(f"Render worker {worker_id} error rendering frame {idx}: {e}", exc_info=True)
            frame_queue.put((idx, canvas))

    logging.debug(f"Render worker {worker_id} exiting.")


def _write_video_worker(frame_queue: multiprocessing.Queue,
                        file_path: str,
                        fourcc_str: str,
                        fps: float,
                        size: tuple,
                        total_frames: int,
                        num_render_workers: int):
    """
    Worker process function to write frames to a video file.
    Retrieves rendered canvases from frame_queue and writes them in order.
    """
    logging.debug("Writer worker started.")
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out = cv2.VideoWriter(file_path, fourcc, fps, (size[0], size[1]))

    frame_buffer = {}
    expected_frame_idx = 0
    frames_written = 0
    sentinels_received = 0

    # Use total_frames if > 0, otherwise tqdm handles total=None (no percentage)
    progress_bar_total = total_frames if total_frames > 0 else None
    progress_bar = tqdm(desc=f"Writing video {file_path}", total=progress_bar_total, unit="frame", smoothing=0.1)

    while True:
        try:
            item = frame_queue.get(timeout=0.1)
        except multiprocessing.queues.Empty: # More specific exception
            # Condition to break: all workers signaled done, buffer is empty, and all expected frames (if known) are processed
            if sentinels_received == num_render_workers and not frame_buffer:
                if total_frames > 0 and expected_frame_idx >= total_frames: # If total_frames is known
                    logging.debug("Writer: All workers done, buffer empty, expected frames processed after timeout.")
                    break
                elif total_frames == 0: # If total_frames was unknown from start
                     logging.debug("Writer: All workers done, buffer empty after timeout (total_frames unknown).")
                     break
            continue

        if item is None:
            sentinels_received += 1
            logging.debug(f"Writer received sentinel. Total sentinels: {sentinels_received}/{num_render_workers}")
            if sentinels_received == num_render_workers and not frame_buffer:
                 if total_frames > 0 and expected_frame_idx >= total_frames:
                    logging.debug("Writer: All workers done, buffer empty, expected frames processed after sentinel.")
                    break
                 elif total_frames == 0:
                     logging.debug("Writer: All workers done, buffer empty after receiving sentinel (total_frames unknown).")
                     break
            continue

        idx, canvas = item
        frame_buffer[idx] = canvas

        while expected_frame_idx in frame_buffer:
            current_canvas = frame_buffer.pop(expected_frame_idx)
            out.write(current_canvas)
            progress_bar.update(1)
            frames_written += 1
            expected_frame_idx += 1
        
        # Additional break condition if all frames are processed and all sentinels received
        if sentinels_received == num_render_workers and not frame_buffer:
            if total_frames > 0 and expected_frame_idx >= total_frames:
                logging.debug(f"Writer: All workers done, buffer empty. Expected {expected_frame_idx}, Total {total_frames}")
                break
            elif total_frames == 0 and expected_frame_idx > 0: # If unknown total, break if workers done and buffer empty.
                logging.debug(f"Writer: All workers done, buffer empty. Expected {expected_frame_idx} (total unknown at start)")
                break


    # Write any remaining frames if producer produced more than approx_length
    while expected_frame_idx in frame_buffer:
        current_canvas = frame_buffer.pop(expected_frame_idx)
        out.write(current_canvas)
        progress_bar.update(1)
        frames_written += 1
        expected_frame_idx += 1
    
    if progress_bar.total is None or progress_bar.total < frames_written : # Update total if it was an estimate or unknown
        progress_bar.total = frames_written
        progress_bar.refresh()

    progress_bar.close()
    out.release()
    logging.debug(f"Writer worker finished. Total frames written: {frames_written}")


class VideoAnnotationWriter:

    def __init__(self, node: RenderNode) -> None:
        """
        Initializes the VideoAnnotationWriter with a RenderNode.

        :param node: The RenderNode to write annotations for.
        """
        self.node = node

    def write_to_file(self, file_path: str, num_render_workers: int = -1, use_multiprocessing: bool = True) -> None:
        """
        Writes the layout and annotations of the RenderNode to a video file.

        :param file_path: The path where the video file will be saved.
        :param num_render_workers: Number of worker processes for rendering (if use_multiprocessing is True).
                                   Defaults to cpu_count() - 1, or 1 if cpu_count() is 1.
        :param use_multiprocessing: If True, uses multiprocessing for rendering. Otherwise, runs on the main thread.
        """
        layout = self.node.get_layout()
        renderers = self.node.renderers
        size = self.node.size  # (width, height)
        fps = self.node.fps
        fourcc_str = 'mp4v' # Common codec

        approx_length = 0 # Default to 0 if not determinable
        try:
            # Try to determine approx_length, but handle cases where it might not be available
            possible_lengths = []
            for renderer_key in renderers: # Iterate through keys to ensure renderer object exists
                renderer = renderers.get(renderer_key)
                if renderer and hasattr(renderer, 'data_streamer') and \
                   hasattr(renderer.data_streamer, 'approx_length') and \
                   renderer.data_streamer.approx_length is not None:
                    possible_lengths.append(renderer.data_streamer.approx_length)
            
            if possible_lengths:
                approx_length = min(possible_lengths)
            else:
                 # Check if self.node itself has an approx_length
                if hasattr(self.node, 'approx_length') and self.node.approx_length is not None:
                    approx_length = self.node.approx_length

            if approx_length == float('inf') or approx_length < 0: # Reset if inf or invalid
                approx_length = 0

        except Exception as e:
            logging.warning(f"Could not determine approximate length of video due to an error: {e}. Progress bar may be inaccurate or show iterations only.")
            approx_length = 0
        
        if approx_length == 0:
            logging.warning("Could not determine a valid approximate length for the video. Progress bar will show iterations without percentage/ETA.")

        if use_multiprocessing:
            self._write_to_file_multiprocessed(file_path, num_render_workers, layout, renderers, size, fps, approx_length, fourcc_str)
        else:
            self._write_to_file_single_threaded(file_path, layout, renderers, size, fps, approx_length, fourcc_str)

    def _write_to_file_single_threaded(self, file_path: str, layout: list, renderers: dict, size: tuple, fps: float, approx_length: int, fourcc_str: str):
        """Helper method for single-threaded writing."""
        logging.info(f"Writing layout to {file_path} (single-threaded).")
        logging.info(f"Output video: {size[0]}x{size[1]} @ {fps}fps, Approx. Frames: {approx_length if approx_length > 0 else 'Unknown'}")

        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(file_path, fourcc, fps, (size[0], size[1]))

        progress_bar_total = approx_length if approx_length > 0 else None
        progress_bar = tqdm(desc=f"Processing video {file_path} (single-thread)", total=progress_bar_total, unit="frame", smoothing=0.1)
        
        actual_frames_written = 0
        try:
            idx = 0
            while True:
                logging.debug(f"Main thread: about to call self.node.next() for frame {idx}")
                data_dict = self.node.next() # This is the source of data
                logging.debug(f"Main thread: got data for frame {idx}")

                canvas_height, canvas_width = size[1], size[0]
                canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

                if data_dict is not None:
                    try:
                        for renderer_id, bbox in layout:
                            if renderer_id in renderers:
                                if renderer_id in data_dict:
                                    renderer_payload = data_dict[renderer_id]
                                    if renderer_payload is not None:
                                        try:
                                            # Assuming data_dict[renderer_id] is (metadata, actual_data)
                                            _, data_for_renderer = renderer_payload
                                            renderers[renderer_id].render(data_for_renderer, bbox, canvas)
                                        except (TypeError, ValueError) as e_unpack:
                                            logging.warning(f"Main thread: Data for renderer {renderer_id} at frame {idx} not in expected (metadata, data) format (Error: {e_unpack}). Payload: {renderer_payload}. Attempting to use payload directly.")
                                            try:
                                                renderers[renderer_id].render(renderer_payload, bbox, canvas)
                                            except Exception as e_direct_render:
                                                logging.error(f"Main thread: Failed to render with direct data for {renderer_id} at frame {idx}: {e_direct_render}", exc_info=True)
                                        except Exception as e_render:
                                            logging.error(f"Main thread: Error rendering specific part for {renderer_id} at frame {idx}: {e_render}", exc_info=True)
                                    else:
                                        logging.debug(f"Main thread: Data for renderer {renderer_id} is None at frame {idx}")
                                else:
                                    logging.debug(f"Main thread: No data key for renderer {renderer_id} in data_dict for frame {idx}")
                            else:
                                logging.debug(f"Main thread: Renderer ID {renderer_id} from layout not found in provided renderers for frame {idx}")
                    except Exception as e_render_loop:
                        logging.warning(f"Main thread: Error during rendering loop for frame {idx}: {e_render_loop}", exc_info=True)
                else:
                     logging.debug(f"Main thread: data_dict is None for frame {idx}. Rendering a blank canvas.")

                out.write(canvas)
                progress_bar.update(1)
                actual_frames_written += 1
                idx += 1

        except StopIteration:
            logging.debug(f"Main thread: StopIteration after {actual_frames_written} frames.")
        except Exception as e_main_loop:
            logging.error(f"Main thread: Unexpected error in video writing loop: {e_main_loop}", exc_info=True)
        finally:
            if progress_bar.total is None or progress_bar.total < actual_frames_written :
                progress_bar.total = actual_frames_written
            progress_bar.n = actual_frames_written # Ensure progress bar shows the correct number of processed frames
            progress_bar.refresh()
            progress_bar.close()
            out.release()
            logging.info(f"Successfully wrote video to {file_path} with {actual_frames_written} frames (single-threaded).")

    def _write_to_file_multiprocessed(self, file_path: str, num_render_workers: int, layout: list, renderers: dict, size: tuple, fps: float, approx_length: int, fourcc_str: str):
        """Helper method for multiprocessed writing."""
        if num_render_workers == -1:
            cpu_cores = multiprocessing.cpu_count()
            num_render_workers = max(1, cpu_cores - 1 if cpu_cores else 1)

        logging.info(f"Writing layout to {file_path} with {num_render_workers} render workers (multiprocessing).")
        logging.info(f"Output video: {size[0]}x{size[1]} @ {fps}fps, Approx. Frames: {approx_length if approx_length > 0 else 'Unknown'}")

        queue_max_size = num_render_workers * 3
        data_queue = multiprocessing.Queue(maxsize=queue_max_size)
        frame_queue = multiprocessing.Queue(maxsize=queue_max_size)

        writer_process = multiprocessing.Process(
            target=_write_video_worker,
            args=(frame_queue, file_path, fourcc_str, fps, size, approx_length, num_render_workers)
        )
        writer_process.start()

        render_workers = []
        for i in range(num_render_workers):
            worker = multiprocessing.Process(
                target=_render_frame_worker,
                args=(data_queue, frame_queue, layout, renderers, size, i)
            )
            render_workers.append(worker)
            worker.start()

        producer_progress_bar_total = approx_length if approx_length > 0 else None
        producer_progress_bar = tqdm(desc=f"Fetching data for {file_path}", total=producer_progress_bar_total, unit="frame", smoothing=0.1)
        
        actual_frames_produced = 0
        try:
            idx = 0
            while True: # Loop indefinitely until self.node.next() raises StopIteration
                logging.debug(f"Producer: about to call self.node.next() for frame {idx}")
                # If approx_length is known and we've already fetched that many,
                # we could potentially stop early, but self.node.next() is the true sentinel.
                data_dict = self.node.next()
                logging.debug(f"Producer: got data for frame {idx}")
                data_queue.put((idx, data_dict))
                actual_frames_produced +=1
                idx += 1
                producer_progress_bar.update(1)

        except StopIteration:
            logging.debug(f"Producer: StopIteration after {actual_frames_produced} frames.")
        except Exception as e_producer:
            logging.error(f"Producer: Error fetching data: {e_producer}", exc_info=True)
        finally:
            if producer_progress_bar.total is None or producer_progress_bar.total < actual_frames_produced:
                 producer_progress_bar.total = actual_frames_produced # Adjust total to actual frames produced
            producer_progress_bar.n = actual_frames_produced
            producer_progress_bar.refresh()
            producer_progress_bar.close()

        logging.debug("Producer: Sending sentinels to render workers...")
        for _ in range(num_render_workers):
            data_queue.put(None)

        logging.debug("Producer: Waiting for render workers to join...")
        for worker in render_workers:
            worker.join()
        logging.debug("Producer: Render workers joined.")

        logging.debug("Producer: Waiting for writer process to join...")
        writer_process.join()
        logging.debug("Producer: Writer process joined. Video writing complete.")
        logging.info(f"Successfully wrote video to {file_path} with {actual_frames_produced} frames (multiprocessing).")
