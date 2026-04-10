import os
from tqdm import tqdm
import pandas as pd
import argparse

import requests
import base64

QWEN3_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
QWEN3_API_KEY = "YOUR-OWN-API"

def encode_img(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_caption_qwen3_api(image_path, prompt_text):
    payload = {
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_img(image_path)}"
                        }
                    }
                ]
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {QWEN3_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(QWEN3_API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Qwen3-VL API ERROR] {image_path}: {e}")
        return "[ERROR_IN_QWEN3_API]"

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["qwen2.5", "qwen3api"], default="qwen2.5", help="选择Qwen2.5本地还是Qwen3-VL API")
args = parser.parse_args()

if args.backend == "qwen2.5":
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        "/workspace/models/Qwen2.5-VL-7B-Instruct", dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("/workspace/models/Qwen2.5-VL-7B-Instruct")

VAL_TSV = "/workspace/compare_model/bart/clip/data/train_pairs.tsv"
OUT_TSV = "/workspace/compare_model/bart/clip/data/train_pairs_emotion.tsv"

df = pd.read_csv(VAL_TSV, sep="\t", names=["img_path", "caption"], keep_default_na=False)
grouped = df.groupby("img_path")["caption"].apply(list).reset_index()

if os.path.exists(OUT_TSV):
    existed = pd.read_csv(OUT_TSV, sep="\t", encoding="utf-8")
    done_imgs = set(existed["img_path"].tolist())
    results = existed.values.tolist()
else:
    done_imgs = set()
    results = []

todo_grouped = grouped[~grouped["img_path"].isin(done_imgs)].reset_index(drop=True)

with open(OUT_TSV, "a", encoding="utf-8") as fout:
    if (not os.path.exists(OUT_TSV)) or os.stat(OUT_TSV).st_size == 0:
        fout.write("img_path\tall_original_captions\temotional_caption\n")

    for idx, row in tqdm(todo_grouped.iterrows(), total=len(todo_grouped), desc="Generating captions"):
        img_path = row["img_path"]
        captions = row["caption"]

        captions_for_prompt = " | ".join(captions)
        prompt_text = (
            f"Here are several different descriptions of the image: {captions_for_prompt} "
            f"Please use them as reference to generate a single, more emotional caption for the image. "
            f"Add suitable adjectives or verbs, make it concise and less than 20 words."
        )

        if args.backend == "qwen2.5":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                        },
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=32)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
        else:
            output_text = generate_caption_qwen3_api(img_path, prompt_text)

        record = [img_path, captions_for_prompt, output_text]
        fout.write("\t".join([str(item) for item in record]) + "\n")
        fout.flush()
