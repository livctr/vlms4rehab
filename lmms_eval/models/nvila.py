import argparse
import importlib.util
import json
import os

from pydantic import BaseModel
from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat

import logging
import torch
from typing import List, Optional, Tuple, Union
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from decord import VideoReader, cpu
from lmms_eval.models.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.api.instance import Instance

eval_logger = logging.getLogger("lmms-eval")
# import sys;sys.path.append("llava-video")



# DEFAULT_IMAGE_TOKEN = "<image>"
# IMAGE_TOKEN_INDEX = -200
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"

# from llava.conversation import SeparatorStyle, conv_templates
# #from llava.data.dataset import LazySupervisedDataset
# from llava.mm_utils import (
#     KeywordsStoppingCriteria,
#     get_model_name_from_path,
#     process_images,
#     tokenizer_image_token,
# )
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init



@register_model("nvila")
class NVILA(lmms):
    """
    NVILA Model
    """

    def __init__(
        self,
        pretrained: str = "Efficient-Large-Model/VILA1.5-40b",
        max_frames_num: Optional[int] = 100,
        sampling_strategy: str = "uniform",
        sampling_fps: int = 8,
        overlap_frames_num: int = 0,
        truncation: Optional[bool] = True,
        device: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation=(
            "sdpa" if torch.__version__ >= "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="auto",
        conv_template="hermes-2",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        video_decode_backend="decord",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        import pdb ; pdb.set_trace()


        self.model = llava.load(pretrained)

        clib.default_conversation = clib.conv_templates[conv_template].copy()

        # prompt = [media, text] Called model.generate_content


        # accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        # accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        # if accelerator.num_processes > 1:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     self.device_map = f"cuda:{accelerator.local_process_index}"
        # elif accelerator.num_processes == 1 and device_map == "auto":
        #     self._device = torch.device(device)
        #     self.device_map = device_map
        # else:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        # self.model_name = get_model_name_from_path(pretrained)
        self.max_frames_num = max_frames_num
        self.sampling_strategy = sampling_strategy
        self.sampling_fps = sampling_fps
        self.overlap_frames_num = overlap_frames_num

        # self._config = AutoConfig.from_pretrained(self.pretrained)

        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, model_base=pretrained, model_name=self.model_name, device_map=self.device_map, attn_implementation=attn_implementation)

        self.model.image_processor = self._image_processor

        self._config = self._model.config

        if self._tokenizer.pad_token_id is None:
            if "qwen" in self._tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self._tokenizer.pad_token_id = 151643

        self.video_decode_backend = video_decode_backend
        self.model.eval()
        # self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        # # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        # if accelerator.num_processes > 1:
        #     assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
        #     # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
        #     # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
        #     # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
        #     if accelerator.distributed_type == DistributedType.DEEPSPEED:
        #         kwargs = {
        #             "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
        #             "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
        #         }
        #         AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
        #         eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
        #     if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
        #         self._model = accelerator.prepare(self.model)
        #     else:
        #         self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
        #     self.accelerator = accelerator
        #     if self.accelerator.is_local_main_process:
        #         eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        #     self._rank = self.accelerator.local_process_index
        #     self._world_size = self.accelerator.num_processes
        # elif accelerator.num_processes == 1 and device_map == "auto":
        #     eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
        #     self._rank = 0
        #     self._world_size = 1
        # else:
        #     eval_logger.info(f"Using single device: {self._device}")
        #     self.model.to(self._device)
        #     self._rank = 0
        #     self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def load_video(self, video_path, max_frames_num):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            fps = round(vr.get_avg_fps())
            frame_idx = np.linspace(0, total_frame_num - 2, max_frames_num, dtype=int)
            spare_frames = vr.get_batch(frame_idx).asnumpy()
            return [Image.fromarray(img) for img in spare_frames]
        except Exception as e:
            eval_logger.error(f"Failed to load video {video_path} with error: {e}")
            return [Image.new("RGB", (448, 448), (0, 0, 0))] * max_frames_num

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

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

            import pdb ; pdb.set_trace()
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            prompt = [Video(visuals[0]), contexts]
            conversation = [{"from": "human", "value": prompt}]

            # Prepare multi-modal prompt

            response = self.model.generate_content(prompt, response_format=None)

            # print("Question: ", cur_prompt)
            print("Answer: ", response)
            res.append(response)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
