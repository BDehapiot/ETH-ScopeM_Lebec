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

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec

#%% Initialize ----------------------------------------------------------------

# Paths
dat_path = Path("D:/local_Lebec/data")
mov_paths = [f for f in dat_path.iterdir() if f.is_dir()]

# Random seed
np.random.seed(42)

#%% Inputs --------------------------------------------------------------------

# Feature detection
feat_params = dict(
    maxCorners=2000, 
    qualityLevel=0.001, # 0.001
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

#%% Function : KLT ------------------------------------------------------------

def preprocess_klt(arr):
    avg = np.mean(arr, axis=(1, 2))
    for t, a in enumerate(avg):
        arr[t, ...] /= a
    arr = norm_pct(arr, sample_fraction=0.01)
    arr = (arr * 255).astype("uint8") 
    return arr

def get_klt_data(arr, feat_params, flow_params):
    
    klt_data = {
        "n"      : [],
        "y"      : [], 
        "x"      : [], 
        "dy"     : [], 
        "dx"     : [],
        "norm"   : [], 
        "status" : [], 
        "error"  : [],
        "feat_params" : feat_params,
        "flow_params" : flow_params,
        }

    # Get frame & features (t0)
    img0 = arr[0, ...]
    f0 = cv2.goodFeaturesToTrack(img0, mask=None, **feat_params)

    for t in range(1, arr.shape[0]):
        
        # Get current image
        img1 = arr[t, ...]
        
        # Compute optical flow (between f0 and f1)
        f1, status, error = cv2.calcOpticalFlowPyrLK(
            img0, img1, f0, None, **flow_params
            )
        
        # Format outputs
        error = error.squeeze().astype(float);
        status = status.squeeze().astype(float); 
        f0 = f0.squeeze(); f1 = f1.squeeze()
        f0[f0[:, 0] >= img0.shape[1]] = np.nan
        f0[f0[:, 1] >= img0.shape[0]] = np.nan
        f1[f1[:, 0] >= img1.shape[1]] = np.nan
        f1[f1[:, 1] >= img1.shape[0]] = np.nan
        f1[status == 0] = np.nan
        
        # Measure norm & xy variations
        norm = np.linalg.norm(f1 - f0, axis=1) 
        dx = f1[:, 0] - f0[:, 0]
        dy = f1[:, 1] - f0[:, 1]
            
        # Append klt_data
        if t == 1:
            nan = np.full_like(status, np.nan)
            klt_data["n"].append(np.nan)
            klt_data["y"].append(f0[:, 1])
            klt_data["x"].append(f0[:, 0])
            klt_data["dy"].append(nan)
            klt_data["dx"].append(nan)
            klt_data["norm"].append(nan)
            klt_data["status"].append(nan)
            klt_data["error"].append(nan)
        klt_data["n"].append(np.nansum(f1[:, 1] > 0))  
        klt_data["y"].append(f1[:, 1])
        klt_data["x"].append(f1[:, 0])
        klt_data["dy"].append(dy)
        klt_data["dx"].append(dx)
        klt_data["norm"].append(norm)
        klt_data["status"].append(status)
        klt_data["error"].append(error)
            
        # Update previous frame & features 
        img0 = img1
        f0 = f1.reshape(-1, 1, 2)
        
    return klt_data

def get_klt_display(arr, klt_data):
    
    klt_display = {
        "coords" : np.zeros_like(arr, dtype=bool),
        "tracks" : np.zeros_like(arr, dtype=bool),
        "labels" : np.zeros_like(arr, dtype='uint16'),
        "norms"  : np.zeros_like(arr, dtype=float),
        "errors" : np.zeros_like(arr, dtype=float),
        }

    for t in range(arr.shape[0]):

        # Extract variables   
        y1s = klt_data["y"][t]
        x1s = klt_data["x"][t]
        norms = klt_data["norm"][t]
        errors = klt_data["error"][t]
        labels = np.arange(y1s.shape[0]) + 1
        
        # Remove non valid data
        valid_idx = ~np.isnan(y1s)
        y1s = y1s[valid_idx].astype(int)
        x1s = x1s[valid_idx].astype(int)
        norms = norms[valid_idx]
        errors = errors[valid_idx]
        labels = labels[valid_idx]
        
        # Fill features display arrays
        klt_display["coords"][t, y1s, x1s] = True
        klt_display["labels"][t, y1s, x1s] = labels
        klt_display["norms" ][t, y1s, x1s] = norms
        klt_display["errors"][t, y1s, x1s] = errors
        
        # Fill tracks display arrays
        if t > 0:
            x0s = klt_data["x"][t-1]
            y0s = klt_data["y"][t-1]
            x0s = x0s[valid_idx].astype(int)
            y0s = y0s[valid_idx].astype(int)
            for x0, y0, x1, y1 in zip(x0s, y0s, x1s, y1s):
                rr, cc = line(y0, x0, y1, x1)
                klt_display["tracks"][t,rr,cc] = True
        
    return klt_display

def plot_klt(klt_data):
    
    # Data
    
    num = klt_data["n"]
    tp = len(klt_data["n"])
    nmax = klt_data["feat_params"]["maxCorners"]
    spd = [np.nanmean(klt_data["norm"][t]) for t in range(tp)]
    dy_avg = [np.nanmean(klt_data["dy"][t]) for t in range(tp)]
    dx_avg = [np.nanmean(klt_data["dx"][t]) for t in range(tp)]
    dy_avg_cum = np.nancumsum(dy_avg, axis=0) 
    dx_avg_cum = np.nancumsum(dx_avg, axis=0) 
    
    # Create figure

    fig = plt.figure(figsize=(3, 3), layout="tight")
    gs = GridSpec(2, 2, figure=fig)
    
    # rcParams
       
    mpl.rcParams.update({
    
    "font.family": "Consolas",
    "font.size": 4,
    "axes.labelsize": 6,
    "axes.titlesize": 8,
    "axes.titlepad": 6,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.color": "black",
    "ytick.color": "black",
    
    "axes.linewidth"   : 0.50,
    "xtick.major.width": 0.25, 
    "ytick.major.width": 0.25, 
    "xtick.minor.width": 0.25, 
    "ytick.minor.width": 0.25, 
    
    "savefig.dpi": 300,
    "savefig.transparent": False,
    
    })
    
    # Track number ------------------------------------------------------------

    # Plot
    ax_num = fig.add_subplot(gs[0, 0]) 
    ax_num.plot(num, linewidth=0.5)
    ax_num.axhline(y=nmax, linewidth=0.5, linestyle="--", color="k") 

    # Format
    ax_num.set_title("Track number")
    ax_num.set_ylim(0, nmax * 1.1)
    ax_num.set_ylabel("Number")
    ax_num.set_xlabel("Timepoint")
    
    # Average track speed -----------------------------------------------------

    # Plot
    ax_spd = fig.add_subplot(gs[0, 1]) 
    ax_spd.plot(spd, linewidth=0.5)
    
    # Format
    ax_spd.set_title("Avg. track speed")
    ax_spd.set_ylim(0, np.nanmax(spd) * 1.1)
    ax_spd.set_ylabel("Speed (pix.tp-1)")
    ax_spd.set_xlabel("Timepoint")
    
    # Average dy/dx -----------------------------------------------------------
    
    # Plot
    ax_dyx = fig.add_subplot(gs[1, 0]) 
    ax_dyx.plot(dy_avg, linewidth=0.5, label="dy")
    ax_dyx.plot(dx_avg, linewidth=0.5, label="dx")
    ax_dyx.axhline(y=0, linewidth=0.5, linestyle="--", color="k") 
    
    # Format
    ax_dyx.set_title("Avg. dy/dx")
    ax_dyx.set_ylabel("Speed (pix.tp-1)")
    ax_dyx.set_xlabel("Timepoint")
    ax_dyx.legend(loc="lower left")
    
    # Cumulative average dy/dx ------------------------------------------------

    # Plot
    ax_cyx = fig.add_subplot(gs[1, 1]) 
    ax_cyx.plot(dy_avg_cum, linewidth=0.5, label="cum_dy")
    ax_cyx.plot(dx_avg_cum, linewidth=0.5, label="cum_dx")
    ax_cyx.axhline(y=0, linewidth=0.5, linestyle="--", color="k")
    
    # Format
    ax_cyx.set_title("Cum. avg. dy/dx")
    ax_cyx.set_ylabel("Speed (pix.tp-1)")
    ax_cyx.set_xlabel("Timepoint")
    ax_cyx.legend(loc="lower left")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    mov_idx = 28
    tmax = 75
        
    # -------------------------------------------------------------------------

    # Load
    mov = []
    mov_path = mov_paths[mov_idx]
    for img_path in list(mov_path.glob("*.tif")):
         mov.append(io.imread(img_path))
    mov = np.stack(mov)[:tmax, ...].astype(float)
        
    # preprocess_klt()
    print("preprocess_klt() : ", end="", flush=False)
    t0 = time.time()
    mov_prp = preprocess_klt(mov)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # get_klt_data()
    print("get_klt_data() : ", end="", flush=False)
    t0 = time.time()
    klt_data = get_klt_data(mov_prp, feat_params, flow_params)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # get_klt_display()
    print("get_klt_display() : ", end="", flush=False)
    t0 = time.time()
    klt_display = get_klt_display(mov, klt_data)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # 
    plot_klt(klt_data)
        
    # Display
    viewer = napari.Viewer()
    viewer.add_image(
        mov, name="mov", visible=1,
        opacity=0.75
        )
    viewer.add_image(
        klt_display["coords"], name="coords", visible=1,
        blending='additive'
        )
    viewer.add_image(
        klt_display["tracks"], name="tracks", visible=1,
        blending='additive'
        )
    
#%%
    

    