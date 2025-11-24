import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers import logging
logging.set_verbosity_error()
from transformers.cache_utils import DynamicCache

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64, load_long_video_decord
from lmms_eval.models.model_utils.caching import longest_common_prefix_len

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl")
class Qwen2_5_VL(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_frames_num: int = 32,
        sampling_strategy: str = "uniform",
        sampling_fps: int = 8,
        overlap_frames_num: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"


        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if use_flash_attention_2:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_frames_num = max_frames_num
        self.sampling_strategy = sampling_strategy
        self.sampling_fps = sampling_fps
        self.overlap_frames_num = overlap_frames_num
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

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
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            assert len(batched_visuals) == 1

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

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
                
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            def build_pil_frames(video):
                return [Image.fromarray(frame.astype(np.uint8)) for frame in video]
        
            def build_input_ids(context, pil_frames):
                messages = []
                msg = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",
                     "content": [
                        {"type": "video", "video": pil_frames, "max_pixels": self.max_pixels, "fps": self.sampling_fps},
                        {"type": "text", "text": context} 
                     ]},
                ]
                messages.append(msg)
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                inputs = self.processor(
                    text=text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs
                )
                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)
                return inputs

            videos = load_long_video_decord(
                batched_visuals[0][0],
                self.max_frames_num,
                self.sampling_strategy,
                self.overlap_frames_num,
                self.sampling_fps,
            )

            outputs = []
            for video, start_time_s, end_time_s in videos:
                video_window_outputs = []

                pil_frames = build_pil_frames(video)
                built_inputs = [build_input_ids(context, pil_frames) for context in context_with_multiple_questions_list]

                if len(built_inputs) == 1:
                    prefix_cache = None
                else:
                    _, suffix_lens = longest_common_prefix_len([bi["input_ids"] for bi in built_inputs])
                    prefix_cache = DynamicCache()

                for i, inputs in enumerate(built_inputs):
                    with torch.inference_mode():
                        cont = self.model.generate(
                            **inputs,
                            eos_token_id=self.tokenizer.eos_token_id,
                            pad_token_id=pad_token_id,
                            do_sample=True if gen_kwargs["temperature"] > 0 else False,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            num_beams=gen_kwargs["num_beams"],
                            max_new_tokens=gen_kwargs["max_new_tokens"],
                            use_cache=self.use_cache,
                            past_key_values=prefix_cache,
                        )
                        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                        if self.use_cache and len(built_inputs) > 1 and i < len(built_inputs) - 1:
                            prefix_cache.crop(-suffix_lens[i] - len(generated_ids_trimmed[0]))

                    text_outputs = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
                    video_window_outputs.append(text_outputs)

                del prefix_cache
                torch.cuda.empty_cache()

                # Join the outputs for this video window
                video_window_outputs = " <SEP> ".join(video_window_outputs)
                outputs.append((video_window_outputs, start_time_s, end_time_s))
            
            outputs_print = "\n".join([f"{out[0]} ({out[1]}s - {out[2]}s)" for out in outputs])
            eval_logger.debug(f"Prediction ({batched_visuals[0][0]}): {outputs_print}")
            res.append(outputs)  # Assumed batch size of 1 up in the beginning
            pbar.update(1)

        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
