import sys

sys.path.append('/content/CC-SHAP-VLM-UOS')

from mm_shap_videos import explain_VLM, extract_frames_from_video, compute_mm_score

import argparse

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers import BitsAndBytesConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multimodality analysis.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")

    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf",
    device_map="auto",
    quantization_config=bnb_config)

    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
    raw_frames = extract_frames_from_video(args.video_path)

    conversation = [
        {

          "role": "user",
          "content": [
              {"type": "text", "text": "What is in the video?\n"},
              {"type": "video"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    _, mm_score, _, _ = explain_VLM(prompt, raw_frames, model, processor, max_new_tokens=30)

    print("T-SHAP (text importance from 0 to 1) is " + str(mm_score))
