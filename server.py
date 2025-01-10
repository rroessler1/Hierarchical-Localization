from typing import Union
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from PIL import Image
import numpy as np
import time
import base64
from inference import HLoc, heuristic, get_inliers_per_match
import io
from PIL import Image
from typing import List
from hloc.localize_inloc import create_transform
from scipy.spatial.transform import Rotation as R
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

app.hloc = HLoc(num_pairs=NUM_PAIRS)

app.transformation = np.eye(4)
app.most_inliers = 0
app.avg_translation = 0
app.avg_translation_n = 0

# @app.post("/items/")
# async def create_item(item: Request):
#     print("received Request")

#     try:
#         rawPNGData = base64.b64decode(item.imageData)
#         pil_image = Image.open(io.BytesIO(rawPNGData))

#         image = np.array(pil_image)
#         results = app.hloc.localize_image(image=image)

#         trajectory_global = {"qx": results["q"][0], "qy": results["q"][1], "qz": results["q"][2], "qw": results["q"][3], "tx": results["t"][0], "ty": results["t"][1], "tz": results["t"][2]}
#         trajectory_local = {"qx": item.rot[0], "qy": item.rot[2], "qz": item.rot[1], "qw": item.rot[3], "tx": item.pos[0], "ty": item.pos[2], "tz": item.pos[1]}

#         curr_transformation = create_transform(trajectory_global) @ np.linalg.inv(create_transform(trajectory_local))
#         curr_translation = curr_transformation[:, 3]
#         inliers = get_inliers_per_match(results)
#         inlier_sum = np.sum([inlier[0] for inlier in inliers[:3]])

#         print(curr_transformation)
#         print(inlier_sum)

#         app.avg_translation = (app.avg_translation_n * app.avg_translation + curr_translation) / app.avg_translation_n
#         app.avg_translation_n += 1

#         if np.linalg.norm(curr_translation[:2] - app.avg_translation[:2], 2) < 4:
#             app.transformation = curr_transformation
#             app.most_inliers = inlier_sum

#     except:
#         print("failed localizing")
    
#     pos = np.array([*item.pos, 1])[[0,2,1,3]]
#     transformed_pos = (app.transformation @ pos) [[0,2,1]]

#     response = Response(pos=transformed_pos, rot=[0,0,0,1])

#     return response

app.pos = [0,0,0]
app.rot = [0,0,0,1]

@app.post("/items/")
async def create_item(item: Request):
    print("received Request")

    try:
        rawPNGData = base64.b64decode(item.imageData)
        pil_image = Image.open(io.BytesIO(rawPNGData))
        
        image = np.array(pil_image)
        results = app.hloc.localize_image(image=image)

        app.pos = results["t"][[0,2,1]]
        app.rot = [0,0,0,1]
    except:
        print("failed localizing")
    
    response = Response(pos=app.pos, rot=app.rot)

    return response