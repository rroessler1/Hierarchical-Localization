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

app.session_path = os.path.join("./datasets/validation/", str(round(time.time() * 1000)))
app.image_path = os.path.join(app.session_path, "images")
app.trajectory_file = os.path.join(app.session_path, "local_trajectories.txt")
os.makedirs(app.image_path, exist_ok=False)
with open(app.trajectory_file, 'a') as txt_file:
    txt_file.write("timestamp,device_id,qw,qx,qy,qz,tx,ty,tz\n")

app.pos = [54, -18, 1]
app.rot = [1, 0, 0, 0]

# @app.post("/items/")
# async def create_item(item: Request):
#     print("received Request")

#     # try:
#     rawPNGData = base64.b64decode(item.imageData)
#     pil_image = Image.open(io.BytesIO(rawPNGData))

#     image = np.array(pil_image)
#     results = app.hloc.localize_image(image=image)


#     results_q = results["q"]
#     results_t = results["t"]

#     trajectory_global = {"qx": results["q"][0], "qy": results["q"][1], "qz": results["q"][2], "qw": results["q"][3], "tx": results["t"][0], "ty": results["t"][1], "tz": results["t"][2]}
#     trajectory_local = {"qx": item.rot[0], "qy": item.rot[2], "qz": item.rot[1], "qw": item.rot[3], "tx": item.pos[0], "ty": item.pos[2], "tz": item.pos[1]}

#     print("here")

#     img_name = str(round(time.time() * 1000))
#     pil_image.save(os.path.join(app.image_path, img_name))

#     print("here2")

#     with open(app.trajectory_file, 'a') as txt_file:
#         trajectory_line = f"{img_name},magicLeap,{results_q[3]},{results_q[0]},{results_q[1]},{results_q[2]},{results_t[0]},{results_t[1]},{results_t[2]}"
#         txt_file.write(trajectory_line)

#     print("here3")

#     print("here4")

#     transformation = create_transform(trajectory_global) @ np.linalg.inv(create_transform(trajectory_local))

#     # xzy
#     q = R.from_matrix(transformation[:3,:3]).as_quat(scalar_first=False)[[0,2,1,3]]
#     t = transformation[[0,2,1], 3]

#     np.atan()

#     # projection onto xy-plane
#     q[1] = 0
#     q /= np.linalg.norm(q, ord=2)

#     print("here5")

#     app.pos = t
#     app.rot = q
#     # except:
#     #     print("failed localizing")
    
#     response = Response(pos=app.pos, rot=app.rot)

#     return response


@app.post("/items/")
async def create_item(item: Request):
    print("received Request")

    try:
        rawPNGData = base64.b64decode(item.imageData)
        pil_image = Image.open(io.BytesIO(rawPNGData))
        
        image = np.array(pil_image)
        results = app.hloc.localize_image(image=image)
        
        print(results)

        img_name = str(round(time.time() * 1000))
        pil_image.save(os.path.join(app.image_path, f"{img_name}.jpg"))

        with open(app.trajectory_file, 'a') as txt_file:
            trajectory_line = f"{img_name},magicLeap,{item.rot[3]},{item.rot[0]},{item.rot[1]},{item.rot[2]},{item.pos[0]},{item.pos[1]},{item.pos[2]}\n"
            txt_file.write(trajectory_line)

        app.pos = results["t"][[0,2,1]]
        app.rot = [0,0,0,1]
    except:
        print("failed localizing")
    
    response = Response(pos=app.pos, rot=app.rot)

    return response