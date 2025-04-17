#%% Imports -------------------------------------------------------------------

import cv2
import time
import napari
import warnings
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# functions
from klt import KLT

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.morphology import disk
from skimage.filters.rank import gradient

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
crop_size = 256
dmax = 5
tmax = 50
parallel = True
display = 1

# KLT
replace = 1
feat_params={
    "maxCorners"        : 100,
    "qualityLevel"      : 1e-4,
    "minDistance"       : 3,
    "blockSize"         : 3,
    "useHarrisDetector" : True,
    "k"                 : 0.04,
    }
flow_params={
    "winSize"           : (9, 9),
    "maxLevel"          : 3,
    "criteria"          : (5, 0.01),
    "minEigThreshold"   : 1e-2,
    }

#%% Function : load() ---------------------------------------------------------

def load(mov_path, crop_size=1024):
    
    mov = []
    for img_path in list(mov_path.glob("*.tif")):
        img = io.imread(img_path)
        if crop_size is not None:
            y0 = (img.shape[0] - crop_size) // 2
            x0 = (img.shape[1] - crop_size) // 2
            y1, x1 = y0 + crop_size, x0 + crop_size
            mov.append(img[y0:y1, x0:x1])
        else:
            mov.append(img)
    
    return np.stack(mov)

#%% Function : subpixel_shift() -----------------------------------------------

def subpixel_shift(mov, dy, dx, norm, dmax=5, tmax=20):
    
    global norm_cum
    
    def _shift(img, dy, dx):
        return shift(
            img.copy(), shift=(-dy, -dx), order=2, mode="wrap")
        
    # Remove invalid tracks (> dmaw)
    if dmax is not None:
        invalid = np.where(norm > dmax)
        dx[invalid] = np.nan
        dy[invalid] = np.nan
            
    # Remove invalid tracks (> tmax)
    if tmax is not None:
        dx[tmax:, :] = np.nan
        dy[tmax:, :] = np.nan
    
    # Measure cumulative dx/dy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        dy_avg = np.nanmean(dy, axis=1)
        dx_avg = np.nanmean(dx, axis=1)
        dy_avg_cum = np.nancumsum(dy_avg, axis=0)
        dx_avg_cum = np.nancumsum(dx_avg, axis=0)
    
    # Shift array
    mov_shifted = Parallel(n_jobs=-1)(
        delayed(_shift)(img, dy, dx)
        for (img, dy, dx) in zip(mov, dy_avg_cum, dx_avg_cum)
        )
        
    return np.stack(mov_shifted)

#%% Function : process() ------------------------------------------------------

def process(mov):
    
    mov = mov.astype("float32")
    
    # Correct brightness
    med = np.median(mov, axis=(1, 2))    
    for t, img in enumerate(mov):
        img /= med[t]

    # Subtract median projection
    med = np.median(mov, axis=0)
    for t, img in enumerate(mov):
        img /= med
        
    # # Make absolute intensities
    # mov = np.abs(mov - 1)
    
    # Convert to uint8
    mov = norm_pct(mov, sample_fraction=0.01)
    mov = (mov * 255).astype("uint8")
        
    return mov

#%% Function : batch() --------------------------------------------------------

def batch(paths, crop_size=1024, parallel=True):
        
    def _batch(path):
        
        # Load
        mov = load(path, crop_size=crop_size)
        
        # KLT
        klt = KLT(
            mov, msk=None, replace=replace,
            feat_params=feat_params, 
            flow_params=flow_params,
            )
        
        # Register
        mov = subpixel_shift(
            mov, klt.dy, klt.dx, klt.norm, dmax=dmax, tmax=tmax)
        
        # Process
        mov = process(mov)
        
        # Save
        io.imsave(
            dat_path / (path.stem + "_process.tif"),
            mov, check_contrast=False,
            )
        
    if parallel and isinstance(paths, list):
        Parallel(n_jobs=-1)(delayed(_batch)(path) for path in paths)
    elif isinstance(paths, list):
        for path in paths:
            print(f"{path.stem}")
            _batch(path)
    else:
        print(f"{paths.stem}")
        _batch(paths)
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # batch(mov_paths, crop_size=crop_size, parallel=parallel)
            
#%%
        
    idx = 49
    path = mov_paths[idx]
        
    print(f"{path.stem}")
    
    # Load
    mov = load(path, crop_size=crop_size)
    mov0 = mov.copy()
    
    # Prepare data for KLT
    prp = norm_pct(mov, sample_fraction=0.01)
    prp = (prp * 255).astype("uint8")
    for t, img in enumerate(prp):
        prp[t, ...] = gradient(img, disk(1))
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(prp)
    
    # KLT
    t0 = time.time()
    print("klt : ", end="", flush=False)
    klt = KLT(
        prp, msk=None, replace=replace,
        feat_params=feat_params, 
        flow_params=flow_params,
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    klt.plot()
    # klt.display()
    
    # Register movie
    t0 = time.time()
    print("register : ", end="", flush=False)
    mov = subpixel_shift(
        mov, klt.dy, klt.dx, klt.norm, dmax=5.0, tmax=20)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Process movie
    t0 = time.time()
    print("process : ", end="", flush=False)
    mov = process(mov)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s\n")
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(mov)
    viewer.add_image(mov0)
