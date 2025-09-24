import argparse
import copy

import numpy as np
import torch
from PIL import Image
import shap
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers import BitsAndBytesConfig

import cv2
import logging
from PIL import Image


bnb_config = BitsAndBytesConfig(load_in_8bit=True)



if 'model' not in globals():
    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
    device_map="auto",
    quantization_config=bnb_config
)


processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", use_fast=True)

model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

t_shap_sum = 0

shorter_vids=filter_shorter_videos(from_file=True, coin_json_path="COIN_small.json", max_duration_s=120)
paths, captions, task_names = get_paths_and_captions(shorter_vids)

for k, caption, video_path in zip(range(len(paths)), captions, paths):

  raw_frames = extract_frames_from_video(video_path, every_n=60)
  _, mm_score, _ = explain_VLM(caption, raw_frames, model, processor, max_new_tokens=80)
  print(mm_score)
  t_shap_sum += mm_score

print(f"T-SHAP_c %     : {t_shap_sum/len(paths)*100:.2f}  ")