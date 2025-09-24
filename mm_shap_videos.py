import argparse
import copy
from types import NoneType

import numpy as np
import torch
from PIL import Image
import shap
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers import BitsAndBytesConfig

import cv2
import logging
from PIL import Image

import json
import random


def filter_shorter_videos(from_file, coin_json_path="", json_object=None, max_duration_s=240):
  if from_file:
    with open(coin_json_path) as f:
      data = json.load(f)
  else:
    if json_object is None:
      raise Exception("Expected json object, but none was provided")
    data = json_object
  filtered_items = {
      key: val
      for key, val in data["database"].items()
      if val.get("duration", 0) <= max_duration_s
  }
  data["database"] = filtered_items
  return data

########################

def get_random_dataset_subset(from_file, coin_json_path="", json_object=None, size=30):
  if from_file:
    with open(coin_json_path) as f:
      data = json.load(f)
  else:
    if json_object is None:
      raise Exception("Expected json object, but none was provided")
    data = json_object
  items = list(data["database"].items())
  if size > len(items):
    raise Exception(f"Can not sample {size} elements from {len(items)} items long dataset")
  sample = random.sample(items, size)
  sampled_database = dict(sample)
  data["database"] = sampled_database
  return data

########################

def format_task_name(task_name):
  return "".join([(" "+i.lower() if i.isupper() else i) for i in task_name]).strip()

def format_annotations(task_name, annotations):
  res = f'This video shows how to {format_task_name(task_name)}. First, you '
  for annotation in annotations:
    res += (f'{annotation["label"]}, then ')
  res += ("you are done!\n")
  return res

########################

def get_paths_and_captions(data):
  captions = []
  video_paths = []
  task_names = []
  for vid, meta in data["database"].items():
      task_name = meta["class"]
      video_paths.append(f'videos/{meta["recipe_type"]}/{vid}.mp4')
      task_names.append(task_name)
      captions.append(format_annotations(task_name, meta["annotation"]))

  return video_paths, captions, task_names
def extract_frames_from_video(video_path, every_n=10, max_duration_mins=5):
    """Extracts frames from a video at a regular interval."""
    cap = cv2.VideoCapture(video_path)
    print(cap)
    print(video_path)
    frames = []
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    if duration > max_duration_mins * 60:
        raise ValueError("Video is too long, I'm not doing this")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
        count += 1
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}\n")

    if len(frames) == 0:
        print("Couldn't extract any frames")
    return frames


def compute_mm_score(nb_frames, shap_values):
    """ Compute Multimodality Score. (80% textual, 20% visual, possibly: 0% knowledge). """
    video_contrib = np.abs(shap_values.values[0, :nb_frames, :]).sum()
    text_contrib = np.abs(shap_values.values[0, nb_frames:, :]).sum()
    text_score = text_contrib / (text_contrib + video_contrib)
    return text_score

def explain_VLM(description, raw_frames, model, processor, max_new_tokens=100, p=None):
    """
    This function returns shap_values.
    Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens).
    """

    instruction  = "Now look at the video and tell me how to do that.\n"

    ## We are formatting and tokeinizing our input manually instead of just using processor()
    # and apply_chat_template(), like a sane person normally would.
    # This is because we need to know exact positions  of the description text
    # in the tokenized sequence, so we can only mask the description tokens
    # and never the chat template or instructions.

    ## The format string for llava one vision looks the following way:
    # "<|im_start|>user <video>\n{text}<|im_end|><|im_start|>assistant\n"

    # Besides, the <video> token expands in-place into (number of frames) * 196 + 1 <video> tokens.
    # 196 because: https://huggingface.co/docs/transformers/en/model_doc/llava_onevision

    nb_special_video_tokens = len(raw_frames) * 196 + 1
    fs_1 = '<|im_start|>user '+  '<video>' * nb_special_video_tokens + '\n'
    fs_2 = '<|im_end|><|im_start|>assistant\n'


    tokenizer = processor.tokenizer

    fs_1_tokens = tokenizer(fs_1, return_tensors="pt", padding=True)
    description_tokens = tokenizer(description, return_tensors="pt", padding=True)
    instruction_tokens = tokenizer(instruction, return_tensors="pt", padding=True)
    fs_2_tokens = tokenizer(fs_2, return_tensors="pt", padding=True)


    nb_fs_1_tokens = len(fs_1_tokens["input_ids"][0])
    nb_desc_tokens = len(description_tokens["input_ids"][0])
    nb_instr_tokens = len(instruction_tokens["input_ids"][0])
    nb_fs_2_tokens = len(fs_2_tokens["input_ids"][0])


    input_ids = torch.cat([fs_1_tokens["input_ids"],  description_tokens["input_ids"], instruction_tokens["input_ids"], fs_2_tokens["input_ids"]], dim=1)
    attention_mask = torch.cat([fs_1_tokens["attention_mask"],  description_tokens["attention_mask"], instruction_tokens["attention_mask"], fs_2_tokens["attention_mask"]], dim=1)
    pixel_values = processor.video_processor(raw_frames, return_tensors="pt")["pixel_values_videos"]


    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values_videos": pixel_values.to("cuda", dtype=torch.float16)
    }

    inputs = BatchEncoding(data)
    inputs = inputs.to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1, do_sample=True)

    output_ids = outputs[:, inputs.input_ids.shape[1]:].to('cpu')  # select only the output ids without repeating the input again

    inputs.to('cpu')

    ########################

    def custom_masker(mask, x):
        """
        Shap relevant function.
        It gets a mask from the shap library with truth values about which image and text tokens to mask (False) and which not (True).
        It defines how to mask the text tokens and masks the text tokens. So far, we don't mask the video, but have only defined which frames to mask.
        The video tokens masking happens in get_model_prediction().
        """
        masked_X = x.clone()  # x.shape is (num_permutations, num_frames + caption text length)

        # hopefully we don't need the line below anymore, as we are not masking system tokens and anything important
        #condition = (masked_X == 151647) | (masked_X == 77091) | (masked_X == 151644) | (masked_X == 151645) | (masked_X == 198)

        #indices = torch.nonzero(condition, as_tuple=False)
        #mask[indices[:, 1]] = True
        frames_mask = torch.tensor(mask).unsqueeze(0)
        # set to zero the frames tokens we are going to mask

        frames_mask[:, -nb_desc_tokens:] = True
        masked_X[~frames_mask] = 0  # ~mask !!! to zero

        # mask the text tokens (delete them)
        text_mask = torch.tensor(mask).unsqueeze(0)
        text_mask[:, :nb_frames] = True  # do not do anything to image tokens anymore

        masked_X[~text_mask] = 62

        return masked_X


    ########################


    # x.shape is (num_permutations aka batch size, num_frames + caption text length)

    def get_model_prediction(x):
        """
        Shap relevant function.
        1. Mask the video frames according to the specified patches to mask from the custom masker.
        2. Predict the model output for all combinations of masked image and tokens. This is then further passed to the shap libary.
        """
        print("##########################################")
        print("##########################################")
        print("Here goes the new batch!")

        with (torch.no_grad()):

          result = np.zeros((x.shape[0], output_ids.shape[1]))
          batch_size = x.shape[0]

          frames_ids = x[:, :nb_frames]
          text_ids = x[:, nb_frames:]


          for i in range(batch_size):
            masked_inputs = copy.deepcopy(inputs)

            # masking frames
            for index, token in enumerate(frames_ids[i]):
              if token == 0:
                masked_inputs.pixel_values_videos[:, index, :, :, :] = 0

            # masking text (description)
            for index, token in enumerate(text_ids[i]):
              if token == 62:
                masked_inputs.input_ids[0][nb_fs_1_tokens + index] = 62

            print((masked_inputs.pixel_values_videos[:, :, 0, 0, 0] != 0.0) * 1)


            masked_inputs.pixel_values_videos.to("cuda", torch.float16)
            masked_inputs.to("cuda")

            # generate outputs and logits
            out = model.generate(**masked_inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                output_logits=True, output_scores=True, return_dict_in_generate=True)

            # out_text = processor.tokenizer.batch_decode( out.sequences, skip_special_tokens=True)[0]
            text = processor.tokenizer.decode(
                out.sequences[0].tolist(), skip_special_tokens=True)
            print(text)
            logits = out.logits[0].detach().cpu().numpy()

            # extract only logits corresponding to target sentence ids
            result[i] = logits[0, output_ids]
        return result


    nb_frames = inputs.pixel_values_videos.shape[1]


    # give the frames some token ids and make them negative to distinguish them from text tokens
    frames_token_ids = torch.tensor(range(-1, -nb_frames - 1, -1)).unsqueeze(0)
    caption_tokens = inputs.input_ids[0][nb_fs_1_tokens : -(nb_instr_tokens + nb_fs_2_tokens)].unsqueeze(0)

    print(frames_token_ids)
    print(caption_tokens)

    print("I am about to combine an X")

    # make a combination between frames ans tokens
    X = torch.cat((frames_token_ids, caption_tokens), 1).unsqueeze(1)

    try:
        explainer = shap.Explainer(get_model_prediction, custom_masker, silent=True, max_evals=600)
        shap_values = explainer(X)[0]
    except ValueError:
        try:
            explainer = shap.Explainer(get_model_prediction, custom_masker, silent=True, max_evals=700)
            shap_values = explainer(X)[0]
        except ValueError:
            try:
                explainer = shap.Explainer(get_model_prediction, custom_masker, silent=True, max_evals=800)
                shap_values = explainer(X)[0]
            except ValueError:
                explainer = shap.Explainer(get_model_prediction, custom_masker, silent=True, max_evals=900)
                shap_values = explainer(X)[0]

    if len(shap_values.values.shape) == 2:
        shap_values.values = np.expand_dims(shap_values.values, axis=2)

    mm_score = compute_mm_score(frames_token_ids.shape[1], shap_values)

    return shap_values, mm_score, p

