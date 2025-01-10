# Image-Based Localization with HLoc

This project runs a server to predict image location using the [HLoc](https://github.com/cvg/Hierarchical-Localization) algorithm.

---

## How It Works

    1. Input: The client sends an image to the server.
    2. Processing: The server uses the HLoc algorithm to predict the location based on the input image.
    3. Output: The server sends the location prediction back to the client.

---

## Data

The following directory structure is necessary for the feature extraction and inference

```bash
.
├── datasets
│ ├── dataset_1 # Input images for localization
│ │ ├──  session_1
│ │ │ ├──  processed_data  
│ │ │ │ ├──  images    
│ │ │ │ │ ├── 1.jpg  
│ │ │ │ │ └── ...  
│ │ │ │ └──  depths  
│ │ │ │ │ ├──  1.png    
│ │ │ │ │ └──  ...  
│ │ │ ├──  images.txt  
│ │ │ ├──  depths.txt  
│ │ │ ├──  rigs.txt  
│ │ │ └──  global_trajectories.txt  
│ │ └──  ...  
│ └── ...  
├── outputs  
│ ├── dataset_1  
│ │ ├── Netvlad_features.h5  
│ │ └── SuperGlue_features.h5  
│ └── ...  
```
## Preprocessing

For preprocessing you need to have for every image in your data a depth image and a known global position and rotation. Then do feature extraction using [HLoc](https://github.com/cvg/Hierarchical-Localization) 

---

## Inference

To do inference we provide two files

- inference.py
- server.py

server.py runs a server that does real time prediction of received images and can be run with:
```bash
fastapi run server.py
```

while inference.py is useful for testing and does inference on a directory of images and visualizes.


---

## Requirements

- Python 3.11
- HLoc library and its dependencies
- Fastapi

---