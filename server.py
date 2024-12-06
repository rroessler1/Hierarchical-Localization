from typing import Union
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from PIL import Image
import numpy as np
import time
import base64
from inference import HLoc
import io
from PIL import Image
from typing import List
import os

NUM_PAIRS = 5

class Request(BaseModel):
    imageData: str
    # fx: float
    # fy: float
    # cx: float
    # cy: float
    # local position
    pos: List[float]
    rot: List[float]

class Response(BaseModel):
    # global position
    pos: List[float]
    rot: List[float]

app = FastAPI()

hloc = HLoc(num_pairs=NUM_PAIRS)

@app.post("/items/")
async def create_item(item: Request):
    rawPNGData = base64.b64decode(item.imageData)
    image = np.array(Image.open(io.BytesIO(rawPNGData)))

    results = hloc.localize_image(image=image)

    t = results["t"]
    q = results["q"]

    print(results)
    
    response = Response(pos=t, rot=q)

    return response
