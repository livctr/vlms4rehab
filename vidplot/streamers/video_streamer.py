from typing import Tuple
import cv2
from vidplot.core import DataStreamer, SizedStreamerProtocol


class VideoStreamer(DataStreamer, SizedStreamerProtocol):
    def __init__(
        self,
        name: str,
        path: str,
        backend: str = "opencv",  # one of 'opencv', 'pyav', 'decord'
    ) -> None:
        """
        Initialize a video streamer.

        Args:
            name (str): Name of the video streamer.
            path (str): Path to the video file.
            backend (str): Backend to use for video processing. Options are 'opencv', 'pyav', or 'decord'.
        Raises:
            IOError: If the video file cannot be opened.
            ValueError: If the backend is unsupported or if FPS cannot be determined.
        Raises:
            AssertionError: If the backend is not one of the supported options.
        Raises:
            ImportError: If the required backend library is not installed.
        """

        super().__init__(name=name)

        # Validate backend and stream method
        assert backend in ["opencv", "pyav", "decord"], f"Unsupported backend '{backend}'"
        self.backend = backend

        # Open and inspect video based on backend
        if backend == "opencv":
            import cv2

            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video {path!r}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        elif backend == "pyav":
            import av

            self.container = av.open(path)
            vs = self.container.streams.video[0]
            self._pyav_iter = self.container.decode(vs)
            self.fps = float(vs.average_rate) if vs.average_rate else 0.0
            self.total_frames = vs.frames or 0
            width, height = vs.width, vs.height

        else:  # decord
            from decord import VideoReader, cpu

            self.vr = VideoReader(path, ctx=cpu(0))
            self.fps = self.vr.get_avg_fps()
            self.total_frames = len(self.vr)
            first_frame = self.vr[0]
            width, height = first_frame.shape[1], first_frame.shape[0]

        if self.fps <= 0:
            raise ValueError(f"Could not determine FPS for backend '{backend}'")
    
        self._duration = self.total_frames / self.fps
        self._size = (width, height)
        self._seeked_num = 0

    @property
    def duration(self) -> float:
        """Approximate duration of the video in seconds."""
        return self._duration

    @property
    def size(self) -> Tuple[int, int]:
        """Return (width, height) of the video frames."""
        return self._size
    
    def __next__(self):
        """
        Retrieve the next frame in the video stream.
        Returns:
            Tuple[float, ndarray]: A tuple containing the timestamp (in seconds) and the frame as a NumPy array.
        Raises:
            StopIteration: If there are no more frames to read.
        """
        # Grab frame based on backend
        if self.backend == "opencv":
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        elif self.backend == "pyav":
            try:
                frame = next(self._pyav_iter)  # raises StopIteration when done
                # Convert to RGB ndarray
                frame = frame.to_ndarray(format="rgb24")
            except StopIteration:
                self.container.close()
                raise StopIteration
        else:  # decord
            try:
                frame = self.vr[self._seeked_num]  # Default RGB
            except IndexError:
                raise StopIteration

        ts = self._seeked_num / self.fps
        self._seeked_num += 1
        return ts, frame
