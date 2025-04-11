#%% Imports -------------------------------------------------------------------

import cv2
import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt 

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.draw import line
from skimage.morphology import binary_dilation, disk

#%% Initialize ----------------------------------------------------------------

# Paths
dat_path = Path("D:/local_Lebec/data")
mov_paths = [f for f in dat_path.iterdir() if f.is_dir()]

# Random seed
np.random.seed(42)

#%% Inputs --------------------------------------------------------------------

# Feature detection
feat_params = dict(
    maxCorners=1000, 
    qualityLevel=0.1, # 0.001
    minDistance=3, # 5
    blockSize=3, # 5
	useHarrisDetector=True
    )

# Optical flow
flow_params = dict(
    winSize=(19, 19), # (11, 11)
    maxLevel=3, # 3
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.01) # 5, 0.01
    )

#%% Function(s) ---------------------------------------------------------------

def klt(stack, feat_params, flow_params):
    
    klt_data = {
        "xCoords" : [],
        "yCoords" : [],
        "status"  : [],
        "errors"  : [],
        "dx"      : [],
        "dy"      : [],
        "dist"    : [],
        }

    # Get frame & features (t0)
    frm0 = stack[0, ...]
    f0 = cv2.goodFeaturesToTrack(
        frm0, mask=None, **feat_params
        )

    for t in range(1, stack.shape[0]):
        
        # Get current image
        frm1 = stack[t, ...]
        
        # Compute optical flow (between f0 and f1)
        f1, status, errors = cv2.calcOpticalFlowPyrLK(
            frm0, frm1, f0, None, **flow_params
            )
        
        # Format outputs
        errors = errors.squeeze().astype(float);
        status = status.squeeze().astype(float); 
        f0 = f0.squeeze(); f1 = f1.squeeze()
        f0[f0[:, 0] >= frm0.shape[1]] = np.nan
        f0[f0[:, 1] >= frm0.shape[0]] = np.nan
        f1[f1[:, 0] >= frm1.shape[1]] = np.nan
        f1[f1[:, 1] >= frm1.shape[0]] = np.nan
        f1[status == 0] = np.nan
        
        # Measure x & y variations
        dx = f1[:, 0] - f0[:, 0]
        dy = f1[:, 1] - f0[:, 1]
        
        # Measure distances
        dist = np.linalg.norm(f1 - f0, axis=1) 
            
        # Append klt_data
        if t == 1:
            nan = np.full_like(status, np.nan)
            klt_data["xCoords"].append(f0[:, 0])
            klt_data["yCoords"].append(f0[:, 1])
            klt_data["status"].append(nan)
            klt_data["errors"].append(nan)
            klt_data["dx"].append(nan)
            klt_data["dy"].append(nan)
            klt_data["dist"].append(nan)
            klt_data["dist"].append(nan)
        klt_data["xCoords"].append(f1[:, 0])
        klt_data["yCoords"].append(f1[:, 1])
        klt_data["status"].append(status)
        klt_data["errors"].append(errors)
        klt_data["dx"].append(dx)
        klt_data["dy"].append(dy)
        klt_data["dist"].append(dist)
            
        # Update previous frame & features 
        frm0 = frm1
        f0 = f1.reshape(-1, 1, 2)
        
    return klt_data

def klt_display(stack, klt_data):
    
    # Create empty diplay arrays
    ftsRaw = np.zeros_like(stack, dtype=bool)
    tksRaw = np.zeros_like(stack, dtype=bool)
    ftsLab = np.zeros_like(stack, dtype='uint16')
    ftsdist = np.zeros_like(stack, dtype=float)
    ftsErr = np.zeros_like(stack, dtype=float)

    for t in range(stack.shape[0]):

        # Extract variables   
        x1s = klt_data['xCoords'][t]
        y1s = klt_data['yCoords'][t]
        dist = klt_data['dist'][t]
        errors = klt_data['errors'][t]
        labels = np.arange(x1s.shape[0]) + 1
        
        # Remove non valid data
        valid_idx = ~np.isnan(x1s)
        x1s = x1s[valid_idx].astype(int)
        y1s = y1s[valid_idx].astype(int)
        dist = dist[valid_idx]
        errors = errors[valid_idx]
        labels = labels[valid_idx]
        
        # Fill features display arrays
        ftsRaw[t, y1s, x1s] = True
        ftsLab[t, y1s, x1s] = labels
        ftsdist[t, y1s, x1s] = dist
        ftsErr[t, y1s, x1s] = errors
        
        # Fill tracks display arrays
        if t > 0:
            x0s = klt_data['xCoords'][t-1]
            y0s = klt_data['yCoords'][t-1]
            x0s = x0s[valid_idx].astype(int)
            y0s = y0s[valid_idx].astype(int)
            for x0, y0, x1, y1 in zip(x0s, y0s, x1s, y1s):
                rr, cc = line(y0, x0, y1, x1)
                tksRaw[t,rr,cc] = True

        # Dilate display arrays
        ftsRaw[t,...] = binary_dilation(ftsRaw[t,...])
        ftsLab[t,...] = binary_dilation(ftsLab[t,...])
        ftsdist[t,...] = binary_dilation(ftsdist[t,...])
        ftsErr[t,...] = binary_dilation(ftsErr[t,...])
        
    return ftsRaw, tksRaw, ftsLab, ftsdist, ftsErr

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    mov_idx = 2
    tmax = 30

    # Load
    mov = []
    mov_path = mov_paths[mov_idx]
    for img_path in list(mov_path.glob("*.tif")):
         mov.append(io.imread(img_path))
    mov = np.stack(mov)[:tmax, ...].astype(float)
    
    # Correct brightness
    avg = np.mean(mov, axis=(1, 2))
    for t, a in enumerate(avg):
        mov[t, ...] /= a
    
    # Convert to uint8
    mov = norm_pct(mov, sample_fraction=0.01)
    mov = (mov * 255).astype("uint8") 
    
    # KLT
    print("klt : ", end="", flush=False)
    t0 = time.time()
    klt_data = klt(mov, feat_params, flow_params)
    diplays = klt_display(mov, klt_data)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    ftsRaw = diplays[0] 
    tksRaw = diplays[1]
    ftsLab = diplays[2]
    
#%% 

    from scipy.ndimage import shift
    
    def fractional_shift_3d(arr, dx, dy, order=3, mode='wrap'):
        result = np.empty_like(arr)
        for i in range(arr.shape[0]):
            result[i] = shift(
                arr[i], shift=(dy[i], dx[i]), order=order, mode=mode)
        return result

    dx_avg, dy_avg = [], []
    for dx, dy in zip(klt_data["dx"], klt_data["dy"]):
        dx_avg.append(np.mean(dx))
        dy_avg.append(np.mean(dy))
    dx_avg = np.stack(dx_avg)
    dy_avg = np.stack(dy_avg)
    dx_avg = np.nancumsum(-dx_avg, axis=0)
    dy_avg = np.nancumsum(-dy_avg, axis=0)
    dx_avg[np.isnan(dx_avg)] = 0
    dy_avg[np.isnan(dy_avg)] = 0

    print("shift : ", end="", flush=False)
    t0 = time.time()
    shifted_mov = fractional_shift_3d(mov.astype(float), dx_avg, dy_avg)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    sub_mov = []
    for t in range(1, shifted_mov.shape[0]):
        sub_mov.append(shifted_mov[t, ...] / shifted_mov[0, ...])
    sub_mov = np.stack(sub_mov)
    
#%%

    # Display
    viewer = napari.Viewer()
    # viewer.add_image(mov, opacity=0.5)
    # viewer.add_image(shifted_mov, opacity=0.5)
    viewer.add_image(sub_mov, opacity=0.5, contrast_limits=[0.8, 1.2])
    # viewer.add_image(ftsRaw, blending='additive')
    # viewer.add_image(tksRaw, blending='additive')
    # viewer.add_labels(ftsLab, blending='additive')

 
    