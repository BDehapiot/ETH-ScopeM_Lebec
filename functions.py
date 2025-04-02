#%% Imports -------------------------------------------------------------------

import cv2
import napari
import numpy as np
from joblib import Parallel, delayed

# bdtools
from bdtools.norm import norm_pct

# skimage
from skimage.draw import line
from skimage.morphology import disk
from skimage.filters.rank import gradient

# scipy
from scipy.ndimage import shift

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec

#%% Function : preprocess_klt() -----------------------------------------------

def preprocess_klt(arr):
    
    def _gradient(img):
        return gradient(img.copy(), footprint=disk(1))
    
    arr = norm_pct(arr, sample_fraction=0.01)
    arr = (arr * 255).astype("uint8") 
    prp = Parallel(n_jobs=-1)(
        delayed(_gradient)(img) 
        for img in arr
        )
    
    return np.stack(prp)   

#%% Function : get_klt_data() -------------------------------------------------

def get_klt_data(
        arr, feat_params, flow_params, 
        data2D=False, display=False, plot=False
        ):

    klt_data = {

        "n"      : [],
        "y"      : [], 
        "x"      : [], 
        "dy"     : [], "dy_avg"   : [],
        "dx"     : [], "dx_avg"   : [],
        "norm"   : [], "norm_avg" : [],
        "status" : [], "error"    : [],
        
        "shape"       : arr.shape,
        "feat_params" : feat_params,
        "flow_params" : flow_params,
        
        }
    
    # Preprocessing
    prp = preprocess_klt(arr)
    
    # Get frame & features (t0)
    nT = prp.shape[0]
    img0 = prp[0, ...]
    f0 = cv2.goodFeaturesToTrack(img0, mask=None, **feat_params)

    for t in range(1, nT):
        
        # Get current image
        img1 = prp[t, ...]
        
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
        dy = f1[:, 1] - f0[:, 1]
        dx = f1[:, 0] - f0[:, 0]
            
        # Append klt_data #1
        if t == 1:
            nan = np.full_like(status, np.nan)
            klt_data["n"].append(np.nansum(f0[:, 1] > 0))
            klt_data["y"].append(f0[:, 1])
            klt_data["x"].append(f0[:, 0])
            klt_data["dy"].append(nan)
            klt_data["dx"].append(nan)
            klt_data["norm"].append(nan)
            klt_data["status"].append(nan)
            klt_data["error"].append(nan)
            klt_data["norm_avg"].append(np.nan)
            klt_data["dy_avg"].append(np.nan)
            klt_data["dx_avg"].append(np.nan)
        klt_data["n"].append(np.nansum(f1[:, 1] > 0))  
        klt_data["y"].append(f1[:, 1])
        klt_data["x"].append(f1[:, 0])
        klt_data["dy"].append(dy)
        klt_data["dx"].append(dx)
        klt_data["norm"].append(norm)
        klt_data["status"].append(status)
        klt_data["error"].append(error)
        klt_data["norm_avg"].append(np.nanmean(norm))
        klt_data["dy_avg"].append(np.nanmedian(dy))
        klt_data["dx_avg"].append(np.nanmedian(dx))
            
        # Update previous frame & features 
        img0 = img1
        f0 = f1.reshape(-1, 1, 2)
        
    # Append klt_data #2 
    klt_data["dy_avg_cum"] = np.nancumsum(klt_data["dy_avg"], axis=0) 
    klt_data["dx_avg_cum"] = np.nancumsum(klt_data["dx_avg"], axis=0) 
        
    # data2D
    if display:
        data2D = True
    if data2D:
        klt_data["klt_data2D"] = get_klt_data2D(klt_data)
        
    # Diplay
    if display:
        display_klt(arr, klt_data)
    
    # Plot
    if plot:
        plot_klt(klt_data)
    
    return klt_data

#%% Function : get_klt_data2D() -----------------------------------------------

def get_klt_data2D(klt_data):
    
    klt_data2D = {
        "2D_yx"     : np.zeros(klt_data["shape"], dtype=bool),
        "2D_tracks" : np.zeros(klt_data["shape"], dtype=bool),
        "2D_labels" : np.zeros(klt_data["shape"], dtype='uint16'),
        "2D_norms"  : np.zeros(klt_data["shape"], dtype=float),
        }

    for t in range(klt_data["shape"][0]):

        # Extract variables   
        y1s = klt_data["y"][t]
        x1s = klt_data["x"][t]
        norms = klt_data["norm"][t]
        labels = np.arange(y1s.shape[0]) + 1
        
        # Remove non valid data
        valid_idx = ~np.isnan(y1s)
        y1s = y1s[valid_idx].astype(int)
        x1s = x1s[valid_idx].astype(int)
        norms = norms[valid_idx]
        labels = labels[valid_idx]
        
        # Fill features display arrays
        klt_data2D["coords"][t, y1s, x1s] = True
        klt_data2D["labels"][t, y1s, x1s] = labels
        klt_data2D["norms" ][t, y1s, x1s] = norms
        
        # Fill tracks display arrays
        if t > 0:
            x0s = klt_data["x"][t-1]
            y0s = klt_data["y"][t-1]
            x0s = x0s[valid_idx].astype(int)
            y0s = y0s[valid_idx].astype(int)
            for x0, y0, x1, y1 in zip(x0s, y0s, x1s, y1s):
                rr, cc = line(y0, x0, y1, x1)
                klt_data2D["tracks"][t,rr,cc] = True

    return klt_data2D

#%% Function : display_klt() --------------------------------------------------

def display_klt(arr, klt_data):
    
    viewer = napari.Viewer()
    viewer.add_image(
        arr, name="arr", visible=1,
        opacity=0.75
        )
    viewer.add_image(
        klt_data["coords"], name="coords", visible=1,
        blending='additive'
        )
    viewer.add_image(
        klt_display["tracks"], name="tracks", visible=1,
        blending='additive'
        )

#%% Function : plot_klt() -----------------------------------------------------

def plot_klt(klt_data):
    
    # Data
    
    num = klt_data["n"]
    nmax = klt_data["feat_params"]["maxCorners"]
    norm_avg = klt_data["norm_avg"]
    dy_avg = klt_data["dy_avg"]
    dx_avg = klt_data["dx_avg"]
    dy_avg_cum = klt_data["dy_avg_cum"]
    dx_avg_cum = klt_data["dx_avg_cum"]
    
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
    ax_nrm = fig.add_subplot(gs[0, 1]) 
    ax_nrm.plot(norm_avg, linewidth=0.5)
    
    # Format
    ax_nrm.set_title("Avg. track speed")
    ax_nrm.set_ylim(0, np.nanmax(norm_avg) * 1.1)
    ax_nrm.set_ylabel("Speed (pix.tp-1)")
    ax_nrm.set_xlabel("Timepoint")
    
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
    # ax_dyx.legend(loc="lower left")
    
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
    # ax_cyx.legend(loc="lower left")



#%% Function : subpixel_shift() -----------------------------------------------

def subpixel_shift(arr, klt_data, order=2):
    
    def _shift(img, dy, dx, order=order):
        return shift(img.copy(), shift=(-dy, -dx), order=order, mode="wrap")
        
    dy_avg_cum = klt_data["dy_avg_cum"]
    dx_avg_cum = klt_data["dx_avg_cum"]
    
    shifted = Parallel(n_jobs=-1)(
        delayed(_shift)(img, dy, dx)
        for (img, dy, dx) in zip(arr, dy_avg_cum, dx_avg_cum)
        )
        
    return np.stack(shifted)