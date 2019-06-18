from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai import *
from fastai.vision import ImageDataBunch, get_transforms, imagenet_stats, load_learner, open_image, models
import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette(debug=True)
path = Path(".")
preTrainedModel="resnet50_EAP_version6_newPyTorch.pkl"
classes = ["0","90","180","270"]





data = ImageDataBunch.single_from_classes(
    path,
    classes,
    ds_tfms=get_transforms(),
    size=224,
).normalize(imagenet_stats)

learn=load_learner(path,preTrainedModel)

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    """
    Predict function called by either classify_url or by upload
    Returns True class as well as the scores
    Scores are *not* probabilities - use activation function (softmax, sigmoid, etc)
    to convert to probabilities!

    """
    
    img = open_image(BytesIO(bytes))
    _, class_,losses = learn.predict(img) 
    return JSONResponse({
        "prediction": classes[class_.item()],
        "scores": sorted(
            zip(learn.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <h3> Image orientation prediction app <h3>
        <h4> Trained on approximately 650 images, each with four representations<h4>
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=80)
