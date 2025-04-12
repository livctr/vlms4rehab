from collections import defaultdict
from typing import List, Optional, Tuple, Union

from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

import llava
from llava import conversation as clib
from llava.constants import MEDIA_TOKENS
from llava.mm_utils import process_images
from llava.utils.tokenizer import tokenize_conversation

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.api.instance import Instance
from lmms_eval.models.model_utils.load_video import load_long_video_decord


@register_model("nvila")
class NVILA(lmms):
    """
    NVILA Model
    """
    def __init__(
        self,
        pretrained: str = "Efficient-Large-Model/NVILA-15B",
        max_frames_num: Optional[int] = 32,
        sampling_strategy: str = "uniform",
        sampling_fps: int = 8,
        overlap_frames_num: int = 0,
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation="flash-attn",
        conv_template="auto",
        video_decode_backend="decord",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        assert attn_implementation == "flash-attn", "Only flash-attn is supported now"
        assert batch_size == 1 or batch_size == '1', "Only batch size 1 is supported now"
        assert video_decode_backend == "decord", "Only video inference and decord is supported now"
        assert conv_template == "auto", "Only auto conversation template is supported now"

        # Load model
        self.model = llava.load(pretrained)

        # Set conversation mode
        clib.default_conversation = clib.conv_templates[conv_template].copy()

        self.pretrained = pretrained
        self.max_frames_num = max_frames_num
        self.sampling_strategy = sampling_strategy
        self.sampling_fps = sampling_fps
        self.overlap_frames_num = overlap_frames_num

        self.video_decode_backend = video_decode_backend
        # self.model.eval()

    # @property
    # def config(self):
    #     # return the associated transformers.AutoConfig for the given pretrained model.
    #     return self._config

    # @property
    # def tokenizer(self):
    #     return self._tokenizer

    # @property
    # def eot_token_id(self):
    #     # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
    #     return self.tokenizer.eos_token_id

    # @property
    # def max_length(self):
    #     return self._max_length

    # @property
    # def batch_size(self):
    #     return self.batch_size_per_gpu

    # @property
    # def device(self):
    #     return self._device

    # @property
    # def rank(self):
    #     return self._rank

    # @property
    # def world_size(self):
    #     return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:

            # encode, pad, and truncate contexts for this batch
            visual = doc_to_visual(self.task_dict[task][split][doc_id])[0]

            videos = load_long_video_decord(
                visual,
                max_frames_num=self.max_frames_num,
                sampling_strategy=self.sampling_strategy,
                overlap_frames_num=self.overlap_frames_num,
                sampling_fps=self.sampling_fps,
                force_sample=False
            )

            outputs = []
            for video in videos:

                images = [Image.fromarray(frame) for frame in video]
                media = {'video': [images]}
                conversation = [{"from": "human", "value": MEDIA_TOKENS['video'] + contexts}]
                media['video'] = [process_images(images,
                                                 self.model.vision_tower.image_processor,
                                                 self.model.config).half()
                                    for images in media['video']]
                input_ids = tokenize_conversation(conversation, self.model.tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)

                media_config = defaultdict(dict)
                generation_config = self.model.default_generation_config
                generation_config.do_sample = False
                generation_config.max_new_tokens = 100

                output_ids = self.model.generate(
                    input_ids=input_ids,
                    media=media,
                    media_config=media_config,
                    generation_config=generation_config,
                    logits_processor=None,
                )

                response = self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

                outputs.append(response)

            eval_logger.debug(f"Context: {contexts}")
            outputs_print = "\n".join(outputs)
            eval_logger.debug(f"NVILA Response: {outputs_print}")
            res.append(outputs)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
