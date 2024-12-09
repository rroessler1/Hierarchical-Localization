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
import uuid
from hloc.localize_inloc import create_transform
from scipy.spatial.transform import Rotation as R


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
    pil_image = Image.open(io.BytesIO(rawPNGData))

    pil_image.save(f"datasets/ml2/{uuid.uuid1()}.jpg")

    image = np.array(pil_image)
    results = hloc.localize_image(image=image)

    print(results)

    trajectory_global = {"qx": results["q"][0], "qy": results["q"][1], "qz": results["q"][2], "qw": results["q"][3], "tx": results["t"][0], "ty": results["t"][1], "tz": results["t"][2]}
    trajectory_local = {"qx": item.rot[0], "qy": item.rot[2], "qz": item.rot[1], "qw": item.rot[3], "tx": item.pos[0], "ty": item.pos[2], "tz": item.pos[1]}

    transformation = create_transform(trajectory_global) @ np.linalg.inv(create_transform(trajectory_local))

    # xzy
    q = R.from_matrix(transformation[:3,:3]).as_quat(scalar_first=False)[[0,2,1,3]]
    t = transformation[[0,2,1], 3]

    # projection onto xy-plane
    q[1] = 0
    q /= np.linalg.norm(q, ord=2)
    
    response = Response(pos=t, rot=q)

    response = Response(pos=[54, -18, 1], rot=[1, 0, 0, 0])

    print(response)

    return response
