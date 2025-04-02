#%% Imports -------------------------------------------------------------------

import cv2
import napari
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# functions
from klt import KLT

# scipy
from scipy.ndimage import shift

#%% Initialize ----------------------------------------------------------------

# Paths
dat_path = Path("D:/local_Lebec/data")
mov_paths = [f for f in dat_path.iterdir() if f.is_dir()]

# Random seed
np.random.seed(42)

#%% Inputs --------------------------------------------------------------------

# Parameters
crop_size = 1024

# Feature detection
feat_params={
    "maxCorners"        : 100,
    "qualityLevel"      : 1e-3,
    "minDistance"       : 3,
    "blockSize"         : 3,
    "useHarrisDetector" : True,
    "k"                 : 0.04,
    }

# Optical flow
flow_params={
    "winSize"           : (9, 9),
    "maxLevel"          : 3,
    "criteria"          : (5, 0.01),
    "minEigThreshold"   : 1e-4,
    }

#%% Function : load() ---------------------------------------------------------

def load(mov_path, crop_size=1024):
    
    mov = []
    for img_path in list(mov_path.glob("*.tif")):
        img = io.imread(img_path)
        y0 = (img.shape[0] - crop_size) // 2
        x0 = (img.shape[1] - crop_size) // 2
        y1, x1 = y0 + crop_size, x0 + crop_size
        mov.append(img[y0:y1, x0:x1])
    
    return np.stack(mov)

#%% Function : subpixel_shift() -----------------------------------------------

def subpixel_shift(arr, klt_data):
    
    def _shift(img, dy, dx):
        return shift(img.copy(), shift=(-dy, -dx), order=2, mode="wrap")
        
    dy_avg_cum = klt_data["dy_avg_cum"]
    dx_avg_cum = klt_data["dx_avg_cum"]
    
    shifted = Parallel(n_jobs=-1)(
        delayed(_shift)(img, dy, dx)
        for (img, dy, dx) in zip(arr, dy_avg_cum, dx_avg_cum)
        )
        
    return np.stack(shifted)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
        
    # Load
    mov = load(mov_paths[1], crop_size=crop_size)
    
    # KLT
    klt = KLT(
        mov, mask=None, 
        feat_params=feat_params, 
        flow_params=flow_params,
        )
    klt.plot()
    
    # 
    y = np.stack(klt.y)
    
    # klt.display()
