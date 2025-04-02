#%% Imports -------------------------------------------------------------------

import cv2
import napari
import warnings
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

#%% Comments ------------------------------------------------------------------

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

'''

Feature Detection Parameters
----------------------------

    maxCorners : int
        Maximum number of features to detect. 
        If more corners exist, only the strongest are returned.
    
    qualityLevel : float
        Minimum accepted quality of features (as a fraction of the best feature). 
        Lower values allow more features.
    
    minDistance : int
        Minimum Euclidean distance between detected features to avoid clustering.
    
    blockSize : int
        Size of the neighborhood (in pixels) used for computing the feature
        quality.
    
    useHarrisDetector : bool
        Indicates whether to use the Harris feature detection method instead of
        the default Shi-Tomasi.
    
    k : float
        Free parameter for the Harris detector.
        Controls the sensitivity of the feature detection
        (commonly between 0.04 and 0.06).

Optical Flow Parameters
-----------------------

    winSize : tuple of int
        Size of the search window at each pyramid level.
        Defines the patch size used to track features between frames.
    
    maxLevel : 3
        Maximum number of pyramid levels to use.
        0 means only the original image is used.
    
    criteria ((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.01)):
        Termination criteria for the iterative search algorithm.
        stops after 5 iterations or when the change is below 0.01.
    
    flags (cv2.OPTFLOW_LK_GET_MIN_EIGENVALS):
        Instructs the algorithm to return the minimum eigenvalue of the gradient 
        matrix as a quality measure instead of the usual tracking error.
    
    minEigThreshold (1e-2):
        Minimum eigenvalue threshold. Features with a value below this are 
        rejected as they are considered too weak for reliable tracking.

'''

#%% Class : KLT ---------------------------------------------------------------

class KLT:
        
    def __init__(
            
            self, arr, mask=None,
            
            feat_params={
                "maxCorners"        : 100,
                "qualityLevel"      : 1e-3,
                "minDistance"       : 3,
                "blockSize"         : 3,
                "useHarrisDetector" : True,
                "k"                 : 0.04,
                }, 
            
            flow_params={
                "winSize"         : (9, 9),
                "maxLevel"        : 3,
                "criteria"        : (5, 0.01),
                "minEigThreshold" : 1e-4,
                },
            
            ):
        
        # Fetch
        self.arr = arr
        self.mask = mask
        self.feat_params = feat_params
        self.flow_params = flow_params
        self.format_flow_params()
        
        # Initialize
        self.shape = arr.shape
        self.nT, self.nY, self.nX = arr.shape
        
        # Procedure
        self.preprocess()
        self.process()
        
    def format_flow_params(self):
                
        criteria = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
        
        self.flow_params = {
            "winSize"         : self.flow_params["winSize"],
            "maxLevel"        : self.flow_params["maxLevel"],
            "criteria"        : (criteria, *self.flow_params["criteria"]),
            "flags"           : cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            "minEigThreshold" : self.flow_params["minEigThreshold"],
            }
        
        return 
        
#%% Method : preprocess() -----------------------------------------------------
        
    def preprocess(self, gradient_radius=1):

        def _gradient(img):
            return gradient(img.copy(), footprint=disk(gradient_radius))
        
        arr = norm_pct(self.arr, sample_fraction=0.01)
        arr = (arr * 255).astype("uint8") 
        self.prp = np.stack(
            Parallel(n_jobs=-1)(delayed(_gradient)(img) for img in arr))
        
#%% Method : process() --------------------------------------------------------
        
    def process(self):
        
        self.n      = []
        self.y      = []
        self.x      = []
        self.dy     = []
        self.dx     = []
        self.norm   = []
        self.status = []
        self.error  = []
                
        # Get frame & features (t0)
        img0 = self.prp[0, ...]
        f0 = cv2.goodFeaturesToTrack(img0, mask=self.mask, **self.feat_params)

        for t in range(1, self.nT):
            
            # Get current image
            img1 = self.prp[t, ...]
            
            # Compute optical flow (between f0 and f1)
            f1, status, error = cv2.calcOpticalFlowPyrLK(
                img0, img1, f0, None, **self.flow_params
                )
            
            # Format outputs
            error = error.squeeze().astype(float)
            status = status.squeeze().astype(float)
            f0 = f0.squeeze(); f1 = f1.squeeze()
            f0[f0[:, 0] >= img0.shape[1]] = np.nan
            f0[f0[:, 1] >= img0.shape[0]] = np.nan
            f1[f1[:, 0] >= img1.shape[1]] = np.nan
            f1[f1[:, 1] >= img1.shape[0]] = np.nan
            f1[status == 0] = np.nan
            
            # Measure norm & xy variations
            dy = f1[:, 1] - f0[:, 1]
            dx = f1[:, 0] - f0[:, 0]
            norm = np.linalg.norm(f1 - f0, axis=1) 
                
            # Append klt data
            if t == 1:
                nan = np.full_like(status, np.nan)
                self.n.append(np.nansum(f0[:, 1] > 0))
                self.y.append(f0[:, 1])
                self.x.append(f0[:, 0])
                self.dy.append(nan)
                self.dx.append(nan)
                self.norm.append(nan)
                self.status.append(nan)
                self.error.append(nan)
            self.n.append(np.nansum(f1[:, 1] > 0))  
            self.y.append(f1[:, 1])
            self.x.append(f1[:, 0])
            self.dy.append(dy)
            self.dx.append(dx)
            self.norm.append(norm)
            self.status.append(status)
            self.error.append(error)
                
            # Update previous frame & features 
            img0 = img1
            f0 = f1.reshape(-1, 1, 2)
            
#%% Method : filter_tracks() --------------------------------------------------

    # def filter_tracks(
    #     self,
    #     min_speed=None,
    #     max_speed=None,
    #     min_length=None,
    #     max_length=None,
    #     ):
        
    #     valid = self.
        
        
            
#%% Method : get_stats() ------------------------------------------------------

    def get_stats(self):
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            self.dy_avg = np.nanmean(np.stack(self.dy), axis=1)
            self.dx_avg = np.nanmean(np.stack(self.dx), axis=1)
            self.norm_avg = np.nanmean(np.stack(self.norm), axis=1)
            self.error_avg = np.nanmean(np.stack(self.error), axis=1)
            self.dy_avg_cum = np.nancumsum(self.dy_avg, axis=0) 
            self.dx_avg_cum = np.nancumsum(self.dx_avg, axis=0) 

#%% Method : get_maps() -------------------------------------------------------

    def get_maps(self):
        
        self.coords_map = np.zeros(self.shape, dtype=bool)
        self.labels_map = np.zeros(self.shape, dtype="uint16")
        self.speeds_map = np.zeros(self.shape, dtype=float)
        self.tracks_map = np.zeros(self.shape, dtype=bool)
        
        for t in range(self.nT):

            # Extract data  
            y1s = self.y[t]
            x1s = self.x[t]
            labels = np.arange(y1s.shape[0]) + 1
            speeds = self.norm[t]

            # Remove non valid data
            valid_idx = ~np.isnan(y1s)
            y1s = y1s[valid_idx].astype(int)
            x1s = x1s[valid_idx].astype(int)
            labels = labels[valid_idx]
            speeds = speeds[valid_idx]
            
            # Fill maps
            self.coords_map[t, y1s, x1s] = True
            self.labels_map[t, y1s, x1s] = labels
            self.speeds_map[t, y1s, x1s] = speeds
            if t > 0:
                y0s = self.y[t-1]
                x0s = self.x[t-1]
                y0s = y0s[valid_idx].astype(int)
                x0s = x0s[valid_idx].astype(int)
                for x0, y0, x1, y1 in zip(x0s, y0s, x1s, y1s):
                    rr, cc = line(y0, x0, y1, x1)
                    self.tracks_map[t,rr,cc] = True

#%% Method : display() --------------------------------------------------------

    def display(self):
        
        if not hasattr(self, "coords_map"):
            self.get_maps()
        
        viewer = napari.Viewer()
        viewer.add_image(
            self.arr, name="arr", visible=1,
            opacity=0.75
            )
        viewer.add_image(
            self.coords_map, name="coords", visible=1,
            blending='additive'
            )
        viewer.add_image(
            self.tracks_map, name="tracks", visible=1,
            blending='additive'
            )
        
#%% Method : plot() -----------------------------------------------------------

    def plot(self):
        
        # rcParams
           
        mpl.rcParams.update({
        
        # Font
        "font.family"        : "Consolas",
        "axes.titlesize"     : 8,
        "axes.labelsize"     : 6,
        "xtick.labelsize"    : 5,
        "ytick.labelsize"    : 5,
        "legend.fontsize"    : 5,
    
        # Padding
        "axes.titlepad"      : 4,  
        "axes.labelpad"      : 2,  
        "xtick.major.pad"    : 2,  
        "ytick.major.pad"    : 2,          
        
        # Linewidth
        "axes.linewidth"     : 0.5,
        "xtick.major.width"  : 0.5, 
        "ytick.major.width"  : 0.5, 
        "xtick.major.size"   : 2,
        "ytick.major.size"   : 2,
        
        # Saving
        "savefig.dpi"         : 300,
        "savefig.transparent" : False,
        
        })
        
        # Initialize
        self.get_stats()
        nmax = self.feat_params["maxCorners"]
        
        # Create figure
    
        fig = plt.figure(figsize=(4, 4), layout="tight")
        gs = GridSpec(3, 3, figure=fig)
                
        # Track count ---------------------------------------------------------
    
        # Plot
        ax_cnt = fig.add_subplot(gs[0, 0]) 
        ax_cnt.plot(self.n, linewidth=0.5)
        ax_cnt.axhline(y=nmax, linewidth=0.5, linestyle="--", color="k") 
    
        # Format
        ax_cnt.set_title("Track count")
        ax_cnt.set_ylim(0, nmax * 1.1)
        ax_cnt.set_ylabel("Count")
        ax_cnt.set_xlabel("Timepoint")
        
        # Average eigenvalue --------------------------------------------------
    
        # Plot
        ax_err = fig.add_subplot(gs[0, 1]) 
        ax_err.plot(self.error_avg, linewidth=0.5)
        # ax_err.axhline(y=nmax, linewidth=0.5, linestyle="--", color="k") 
    
        # Format
        ax_err.set_title("Avg. eigenvalue")
        ax_err.set_ylim(0, np.nanmax(self.error_avg) * 1.1)
        ax_err.set_ylabel("Eigenvalue")
        ax_err.set_xlabel("Timepoint")
        
        # Average speed -------------------------------------------------------
    
        # Plot
        ax_nrm = fig.add_subplot(gs[1, 0]) 
        ax_nrm.plot(self.norm_avg, linewidth=0.5)
        
        # Format
        ax_nrm.set_title("Avg. speed")
        ax_nrm.set_ylim(0, np.nanmax(self.norm_avg) * 1.1)
        ax_nrm.set_ylabel("Speed (pix.tp-1)")
        ax_nrm.set_xlabel("Timepoint")
        
        # Average dy/dx -------------------------------------------------------
        
        # Plot
        ax_dyx = fig.add_subplot(gs[1, 1]) 
        ax_dyx.plot(self.dy_avg, linewidth=0.5, label="dy")
        ax_dyx.plot(self.dx_avg, linewidth=0.5, label="dx")
        ax_dyx.axhline(y=0, linewidth=0.5, linestyle="--", color="k") 
        
        # Format
        ax_dyx.set_title("Avg. dy/dx")
        ax_dyx.set_ylabel("dy/dx (pix.tp-1)")
        ax_dyx.set_xlabel("Timepoint")
        # ax_dyx.legend(loc="lower left")
        
        # Cumulative average dy/dx --------------------------------------------
    
        # Plot
        ax_cyx = fig.add_subplot(gs[1, 2]) 
        ax_cyx.plot(self.dy_avg_cum, linewidth=0.5, label="cum_dy")
        ax_cyx.plot(self.dx_avg_cum, linewidth=0.5, label="cum_dx")
        ax_cyx.axhline(y=0, linewidth=0.5, linestyle="--", color="k")
        
        # Format
        ax_cyx.set_title("Cum. avg. dy/dx")
        ax_cyx.set_ylabel("Cum. dy/dx (pix.tp-1)")
        ax_cyx.set_xlabel("Timepoint")
        # ax_cyx.legend(loc="lower left")
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    import time
    from skimage import io
    from pathlib import Path

    # -------------------------------------------------------------------------

    def load(mov_path, crop_size=1024):
        
        mov = []
        for img_path in list(mov_path.glob("*.tif")):
            img = io.imread(img_path)
            y0 = (img.shape[0] - crop_size) // 2
            x0 = (img.shape[1] - crop_size) // 2
            y1, x1 = y0 + crop_size, x0 + crop_size
            mov.append(img[y0:y1, x0:x1])
        
        return np.stack(mov)
    
    # -------------------------------------------------------------------------
    
    # Paths
    dat_path = Path("D:/local_Lebec/data")
    mov_paths = [f for f in dat_path.iterdir() if f.is_dir()]

    # Load
    mov = load(mov_paths[1], crop_size=1024)
    
    # KLT
    t0 = time.time()
    klt = KLT(
        mov, mask=None, 
        feat_params=feat_params, 
        flow_params=flow_params,
        )
    t1 = time.time()
    
    klt.plot()
    y, x, dy, dx = klt.y, klt.x, klt.dy, klt.dx
