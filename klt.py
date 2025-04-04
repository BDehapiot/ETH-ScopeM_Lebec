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
    "maxCorners"        : 1000,
    "qualityLevel"      : 1e-4,
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
                
        # 
        
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
            
            # Format data
            error, status, f0, f1 = [
                x.squeeze() for x in (error, status, f0, f1)]

            # Remove "out of frame" features
            def out_of_frame(f):
                x, y = f[:, 0], f[:, 1]
                return (x <= 0) | (x >= self.nX) | (y <= 0) | (y >= self.nY)
            f0[out_of_frame(f0)] = np.nan
            f1[out_of_frame(f1)] = np.nan
            f1[status == 0] = np.nan
            
            # Replace lost features *******************************************
            
            lost_idx = np.where(np.isnan(f1[:, 0]))[0]
            n_lost = len(lost_idx)
            print(t, n_lost)
            
            if n_lost > 0:
                new_mask = self.mask.copy() if self.mask is not None else None
                if new_mask is not None:
                    valid_feats = f1[~np.isnan(f1[:, 0])]
                    for pt in valid_feats:                        
                        cv2.circle(
                            new_mask, 
                            (int(pt[0]), int(pt[1])), 
                            int(self.feat_params["minDistance"]), 0, -1)
                        
                new_feat_params = self.feat_params.copy()
                new_feat_params["maxCorners"] = n_lost
                new_feats = cv2.goodFeaturesToTrack(
                    img1, mask=self.mask, **new_feat_params)
                new_feats = new_feats.squeeze()
            
                self.new_feats = new_feats
                new_feats[out_of_frame(new_feats)] = np.nan
                
                f1[lost_idx] = new_feats
            
            
                # if new_feats is not None:
                #     new_feats = new_feats.squeeze()
                    
                #     if new_feats.ndim == 1:
                #         new_feats = new_feats[np.newaxis, :]
                #     f1[lost_idx] = new_feats
              
            # *****************************************************************
            
            self.f0, self.f1 = f0, f1
            
            # Measure norm & dyx
            dy = f1[:, 1] - f0[:, 1]
            dx = f1[:, 0] - f0[:, 0]
            norm = np.linalg.norm(f1 - f0, axis=1) 
            
            # Append data
            if t == 1:
                zeros = np.full_like(status, 0)
                nan = np.full_like(error, np.nan)
                self.n.append(np.nansum(f0[:, 1] > 0))
                self.y.append(f0[:, 1])
                self.x.append(f0[:, 0])
                self.dy.append(nan)
                self.dx.append(nan)
                self.norm.append(nan)
                self.status.append(zeros)
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
 
#%%
    
    # KLT
    t0 = time.time()
    print("KLT : ", end="", flush=False)
    klt = KLT(
        mov, mask=None, 
        feat_params=feat_params, 
        flow_params=flow_params,
        )
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")

    f0, f1 = klt.f0, klt.f1
    status, error = klt.status, klt.error
    n, y, x = klt.n, klt.y, klt.x
    dy, dx, norm = klt.dy, klt.dx, klt.norm
    
    new_feats = klt.new_feats
