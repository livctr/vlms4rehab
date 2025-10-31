from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from lmms_eval.models.model_utils.load_video import load_long_video_decord


from lmms_eval import utils

from tqdm import tqdm

@register_model("bot")
class Bot(lmms):
    """
    Dummy LlavaVid Model that returns dummy values for all functions.
    This class implements the same interface as LlavaVid but does not perform any real computations.
    """

    def __init__(
        self,
        truncation: Optional[bool] = True,
        device: Optional[str] = "cpu",
        batch_size: Optional[Union[int, str]] = 1,
        sampling_strategy: Optional[str] = "dense",
        sampling_fps: Optional[int] = 8,
        overlap_frames_num: Optional[int] = 0,
        max_frames_num=8,
        video_decode_backend: Optional[str] = "decord",
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

        self.sampling_strategy = sampling_strategy
        self.sampling_fps = sampling_fps
        self.overlap_frames_num = overlap_frames_num
        self.max_frames_num = max_frames_num
        self.video_decode_backend = video_decode_backend

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

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # Return a dummy loglikelihood of 0.0 and True for each request.
        return [(0.0, True) for _ in requests]

    def flatten(self, input):
        return [item for sublist in input for item in sublist]

    def generate_until(self, requests) -> List[str]:
        # Return a dummy response for each request.
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1
            assert self.video_decode_backend == "decord"

            videos = load_long_video_decord(
                batched_visuals[0][0],
                self.max_frames_num,
                self.sampling_strategy,
                self.overlap_frames_num,
                self.sampling_fps,
            )

            def separate_context(context: str):
                if "<SEP>" not in context:
                    return [context]
                # Split the context by <SEP> and remove leading/trailing whitespace
                parts = [part.strip() for part in context.split("<SEP>")]
                # Remove empty parts
                parts = [part for part in parts if part]
                return parts
            context_with_multiple_questions_one_string = batched_contexts[0]
            context_with_multiple_questions_list = separate_context(context_with_multiple_questions_one_string)

            outputs = []
            for video, start_time_s, end_time_s in videos:

                video_window_outputs = []

                for context in context_with_multiple_questions_list:
                    sim_response = "FINAL_ANSWER: 1"
                    video_window_outputs.append(sim_response)
                # Join the outputs for this video window
                video_window_outputs = " <SEP> ".join(video_window_outputs)
                # import pdb ; pdb.set_trace()
                outputs.append((video_window_outputs, start_time_s, end_time_s))

            res.append(outputs)
        
        res = re_ords.get_original(res)
        return res


    def generate_until_multi_round(self, requests):
        return super().generate_until_multi_round(requests)
