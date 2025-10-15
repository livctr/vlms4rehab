from typing import List, Optional, Tuple, Union

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import decord
import numpy as np
import torch
# from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
import hashlib
import pickle
from pathlib import Path

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    # eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")
    pass


class Qwen2_5_VL_VQA:
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=False,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_frames_num: int = 32,
        sampling_strategy: str = "uniform",
        sampling_fps: int = 8,
        overlap_frames_num: int = 0,
        cache_dir: Optional[str] = "/gpfs/data/schambralab/quantitativeRehabilitation/__lab_member_homes/naveen/final_pipeline/the_pipeline/strokerehab/strokerehab/organized_pipeline/hf_home",
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        print("Initializing Qwen2_5_VL_VQA...")
        self._device = torch.device(device)
        self.device_map = device_map

        print(f"Using device: {self._device}")

        # /gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00023/C00023_RTT right side1_1.csv

        attn_impl = "flash_attention_2" if use_flash_attention_2 else None
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            low_mem = True
        else:
            dtype = torch.float32
            low_mem = False  # critical: prevents SLURM deadlock

        model_kwargs = {
            "pretrained_model_name_or_path": pretrained,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": low_mem,
        }
        print(f"Model kwargs: {model_kwargs}")
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl
        if device_map is not None:
            model_kwargs["device_map"] = device_map  # <-- IMPORTANT: honor device_map
        print(f"Loading model {pretrained}...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(**model_kwargs).eval()
        print("Model loaded.")
        self._config = self._model.config
        print(f"Model config: {self._config}")

        # If no device_map was provided, place model on a single device as before
        print("Setting up model device...")
        if device_map is None:
            self._model.to(self._device)

        print("Model device set.")
        if torch.cuda.is_available():
            self._input_device = torch.device("cuda")
        else:
            self._input_device = torch.device("cpu")

        print("Loading processor and tokenizer...")
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        print("Processor loaded.")
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        print("Tokenizer loaded.")
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_frames_num = max_frames_num
        self.sampling_strategy = sampling_strategy
        self.sampling_fps = sampling_fps
        self.overlap_frames_num = overlap_frames_num
        self._pretrained = pretrained

        # self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        self._rank = 0
        self._world_size = 1

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}  # in-memory cache

        print("Qwen2_5_VL_VQA initialized.")

    def _hash_input(self, frames_rgb: List[np.ndarray] | np.ndarray, context: str) -> str:
        """
        Compute a hash key based on frames and context.
        """
        hasher = hashlib.sha256()
        if isinstance(frames_rgb, np.ndarray):
            hasher.update(frames_rgb.tobytes())
        else:
            for frame in frames_rgb:
                hasher.update(frame.tobytes())
        hasher.update(context.encode("utf-8"))
        hasher.update(self._pretrained.encode("utf-8"))
        return hasher.hexdigest()

    def _load_from_disk_cache(self, key: str):
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def _save_to_disk_cache(self, key: str, value: str):
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(value, f)

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

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def clear(self):
        pass
    
    def build_pil_frames(self, frames: List[np.ndarray] | np.ndarray):
        if isinstance(frames, np.ndarray) and len(frames.shape) == 3:
            frames = [frames]
        return [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    
    def build_input_ids(self, context, pil_frames):
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
        # if self.device_map == "auto":
        #     inputs = inputs.to("cuda")
        # else:
        #     inputs = inputs.to(self.device)
        inputs = inputs.to(self._input_device)
        return inputs

    def process_frames(self, frames_rgb: List[np.ndarray] | np.ndarray, context: str, max_new_tokens: int = 256):

        if self.use_cache:
            key = self._hash_input(frames_rgb, context)
            if key in self._cache:
                return self._cache[key]
            disk_val = self._load_from_disk_cache(key)
            if disk_val is not None:
                self._cache[key] = disk_val
                return disk_val

        pil_frames = self.build_pil_frames(frames_rgb)
        inputs = self.build_input_ids(context, pil_frames)

        with torch.inference_mode():
            # Generate deterministically
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]

        text_outputs = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        if self.use_cache:
            self._cache[key] = text_outputs
            self._save_to_disk_cache(key, text_outputs)
        return text_outputs