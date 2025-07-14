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
        raise ValuError("Video is too long, I'm not doing this")

    while cap.isOpened():
        ret, frame = cap.read()
        # if not ret or len(frames) >= max_frames:
        #     break
        if not ret:
            break
        if count % every_n == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
        count += 1
    cap.release()
    # logging.info(f"Extracted {len(frames)} frames from {video_path}")
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


def explain_VLM(prompt, raw_frames, model, processor, max_new_tokens=100, p=None):
    """
    This is the equivalent function of explain_lm. It returns shap_values.
    Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens).
    """

    inputs = processor(text=prompt, videos=[raw_frames], return_tensors='pt').to("cuda", torch.float16)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1, do_sample=True)

    # TODO: check this line
    output_ids = outputs[:, inputs.input_ids.shape[1]:].to(
        'cpu')  # select only the output ids without repeating the input again
    inputs.to('cpu')

    def custom_masker(mask, x):
        """
        Shap relevant function.
        It gets a mask from the shap library with truth values about which image and text tokens to mask (False) and which not (True).
        It defines how to mask the text tokens and masks the text tokens. So far, we don't mask the image, but have only defined which image tokens to mask. The image tokens masking happens in get_model_prediction().
        """
        masked_X = x.clone()  # x.shape is (num_permutations, num_frames+text_length)

        condition = (masked_X == 151647)
        indices = torch.nonzero(condition, as_tuple=False)
        mask[indices[:, 1]] = True
        frames_mask = torch.tensor(mask).unsqueeze(0)
        # set to zero the frames tokens we are going to mask
        frames_mask[:, -nb_text_tokens:] = True  # do not mask text tokens yet
        masked_X[~frames_mask] = 0  # ~mask !!! to zero

        # mask the text tokens (delete them)
        text_mask = torch.tensor(mask).unsqueeze(0)
        text_mask[:, :nb_frames] = True  # do not do anything to image tokens anymore

        masked_X[~text_mask] = 62

        return masked_X

    def get_model_prediction(x):
        """
        Shap relevant function.
        1. Mask the image pixel according to the specified patches to mask from the custom masker.
        2. Predict the model output for all combinations of masked image and tokens. This is then further passed to the shap libary.
        """
        with (torch.no_grad()):
            text_ids = torch.tensor(x[:, -nb_text_tokens:])  # text ids
            masked_frames_token_ids = torch.tensor(x[:, :nb_frames])
            # output_ids.shape is (1, output_length); result.shape is (num_permutations, output_length)
            result = np.zeros((text_ids.shape[0], output_ids.shape[1]))
            # TODO: check this
            batch_size = inputs.input_ids.shape[0]

            for i in range(batch_size):
                # here the actual masking of the image is happening. The custom masker only specified which patches to mask, but no actual masking has happened
                masked_inputs = copy.deepcopy(inputs)  # initialize the thing
                masked_inputs.text_ids = text_ids[i].unsqueeze(0)
                for k in range(masked_frames_token_ids[i].shape[0]):
                    if masked_frames_token_ids[i][k] == 0:
                        masked_inputs.pixel_values_videos[:, k, :, :] = 0

            masked_inputs.to("cuda", torch.float16)
            # # generate outputs and logits
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
    # give the  frames some token ids and make them negative to distinguish them from text tokens
    frames_token_ids = torch.tensor(range(-1, -nb_frames, -1)).unsqueeze(0)

    text_token_mask = inputs.input_ids != model.config.video_token_id
    text_token_ids = inputs.input_ids[text_token_mask].unsqueeze(0)
    nb_text_tokens = text_token_ids.shape[1]

    print("Frames token ids shape: ")
    print(frames_token_ids.shape)

    print("input  ids shape: ")
    print(text_token_ids.shape)

    # make a combination between tokens and frames
    X = torch.cat((frames_token_ids, text_token_ids), 1).unsqueeze(1)
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

    return shap_values, mm_score, p, nb_text_tokens


