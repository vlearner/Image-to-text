import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return "Use /img-text to get image captioning"


@app.post("/img-text/")
async def image_captioning(img_url: str):
    return img_url


