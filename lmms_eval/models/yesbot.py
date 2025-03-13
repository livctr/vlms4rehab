from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

@register_model("yesbot")
class YesBot(lmms):
    """
    Dummy LlavaVid Model that returns dummy values for all functions.
    This class implements the same interface as LlavaVid but does not perform any real computations.
    """

    def __init__(
        self,
        truncation: Optional[bool] = True,
        device: Optional[str] = "cpu",
        batch_size: Optional[Union[int, str]] = 1,
    ) -> None:
        super().__init__()
        # Minimal initialization for the dummy model
        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        self._max_length = 1024
        self._config = {"dummy": True}
        self.conv_template = "dummy"
        self.use_cache = True
        self.truncate_context = truncation
        self._rank = 0
        self._world_size = 1

        # Dummy tokenizer with basic encode/decode functionality
        class DummyTokenizer:
            def encode(self, s, add_special_tokens=False):
                return [1, 2, 3]

            def decode(self, tokens):
                return "dummy_decoded_text"

            @property
            def eos_token_id(self):
                return 0

            @property
            def padding_side(self):
                return "right"

            @property
            def pad_token_id(self):
                return 0

            @property
            def name_or_path(self):
                return "dummy"

        self._tokenizer = DummyTokenizer()
        self._model = None  # No actual model
        self._image_processor = None  # No image processing in the dummy

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        return torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def load_image(self, image_path):
        # Return a list of 10 dummy white images.
        dummy_image = Image.new("RGB", (100, 100), color="white")
        return [dummy_image for _ in range(10)]

    def load_video(self, video_path, max_frames_num, fps, force_sample=False):
        # Return dummy video frames, a frame time string, and a dummy video duration.
        dummy_frames = np.zeros((max_frames_num, 336, 336, 3), dtype=np.uint8)
        dummy_frame_time = ",".join([f"{0.00:.2f}s" for _ in range(max_frames_num)])
        dummy_video_time = max_frames_num / fps if fps != 0 else 0
        return dummy_frames, dummy_frame_time, dummy_video_time

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # Return a dummy loglikelihood of 0.0 and True for each request.
        return [(0.0, True) for _ in requests]

    def flatten(self, input):
        return [item for sublist in input for item in sublist]

    def generate_until(self, requests) -> List[str]:
        # Return a dummy response for each request.
        return ["yes! most definitely yes!" for _ in requests]

    def generate_until_multi_round(self, requests) -> List[str]:
        # Return a dummy multi-round response for each request.
        return ["yes! multi-round yes!" for _ in requests]
