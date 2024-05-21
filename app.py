import uvicorn
from fastapi import FastAPI

from model.blip_image import image_caption, json_response, image_location

app = FastAPI()


@app.get("/")
def read_root():
    return "Use /img-text to get image captioning"


@app.post("/img-text/")
async def blip_image_captioning_large(img_url: str):
    img_raw = image_location(img_url)
    img_caption_1, img_caption_2 = image_caption(img_raw)
    response = json_response(img_caption_1, img_caption_2)
    return response


