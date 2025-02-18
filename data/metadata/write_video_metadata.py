import av
import cv2
from data.utils import write_metadata
import logging

from data.utils_strokerehab import DataPaths


def cnt_frames_av(path):
    container = av.open(path)
    return sum(1 for _ in container.decode(video=0))


def get_codec_av(file_path):
    container = av.open(file_path)
    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    codec = video_stream.codec_context.name if video_stream else None
    return codec


def get_video_info(path):
    try:
        cap = cv2.VideoCapture(path)
        cv2_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Count frames using cv2 while loop
        while_loop_cnt = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            while_loop_cnt += 1
        
        # get count from av
        av_cnt = cnt_frames_av(path)

        cap.release()
        return {
            "path": path,
            "codec": get_codec_av(path),
            "fps": fps,
            "height": height,
            "width": width,
            "duration": cv2_nframes / fps if fps > 0 else None,
            "av_nframes": av_cnt,
            "cv2_nframes": cv2_nframes,
            "cv2_nframes_while_loop": while_loop_cnt,
            "aligned_nframes": cv2_nframes == while_loop_cnt == av_cnt,
        }
    except Exception as e:
        logging.warning(f"Error processing {path}: {e}")
        return None


def write_video_metadata():
    """
    Writes video metadata in folder `DataPaths.RAW_VIDEO_DIR` to
    `DataPaths.VIDEO_METADATA_PATH`.
    """
    write_metadata(DataPaths.RAW_VIDEO_DIR, DataPaths.VIDEO_METADATA_PATH, get_video_info)

