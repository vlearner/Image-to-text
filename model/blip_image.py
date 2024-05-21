from typing import Any

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


def image_location(img_url: str):
    global image_path
    if img_url.startswith('http'):
        print('looking image from the web')
        image_path = Image.open(requests.get(img_url, stream=True, ).raw).convert('RGB')
    elif img_url is None:
        print('No image found')
    else:
        print('Using local image')
        image_path = Image.open(img_url).convert('RGB')
    return image_path


def image_caption(img_raw: Any):
    text = "a photography of"
    inputs = processor(img_raw, text, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=200)
    img_caption_1 = processor.decode(out[0], skip_special_tokens=True)
    inputs = processor(img_raw, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=200)
    img_caption_2 = processor.decode(out[0], skip_special_tokens=True)
    return img_caption_1, img_caption_2


def json_response(img_caption_1, img_caption_2):
    return {"Description": [img_caption_1, img_caption_2]}
