#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt 

# bdtools
from bdtools.models.unet import UNet

# scipy
from scipy.optimize import curve_fit

#%% Inputs --------------------------------------------------------------------

# Parameters
crop_size = 512

# predict
stk_idx = 41
model_name = "model_256_normal_4000-271_1"

#%% Initialize ----------------------------------------------------------------

dat_path = Path("D:\local_Lebec\data")
trn_path = Path("data", "train")
stk_paths = [f for f in dat_path.iterdir() if f.is_dir()]

#%% Function(s) ---------------------------------------------------------------

def load(stk_path, crop_size=1024):
    
    stk = []
    for img_path in list(stk_path.glob("*.tif")):
        img = io.imread(img_path)
        if crop_size is not None:
            y0 = (img.shape[0] - crop_size) // 2
            x0 = (img.shape[1] - crop_size) // 2
            y1, x1 = y0 + crop_size, x0 + crop_size
            stk.append(img[y0:y1, x0:x1])
        else:
            stk.append(img)
    
    return np.stack(stk)

def l5p(x, A, D, C, B, E):
    
    """
    A = lower asymptote
    D = upper asymptote
    C = inflection point (EC50)
    B = slope factor
    E = asymmetry factor (E=1 reduces to 4PL)
    """

    return A + (D - A) / ((1 + np.exp(-B*(x - C)))**E)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
            
    # for stk_path in stk_paths:
        
    # Load
    stk_path = stk_paths[34]
    stk = load(stk_path, crop_size=crop_size)
    
    # Predict
    unet = UNet(load_name=model_name)
    prd = unet.predict(stk, verbose=3)
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(stk)
    viewer.add_image(prd)
    
#%%
    
    # L5P fit -----------------------------------------------------------------
    
    y = np.mean(prd, axis=(1, 2))
    x = np.arange(len(y))
    p0 = [np.min(y), np.max(y), np.median(x), 1.0, 1.0]
    bounds = (
        [0, 0, 0, 0, 0],
        [np.inf, np.inf, np.inf, np.inf, np.inf],
        )
    popt, pcov = curve_fit(l5p, x, y, p0=p0, bounds=bounds)
    fA, fD, fC, fB, fE = popt
    
    
    # Plot --------------------------------------------------------------------
    
    # Initialize
    xf = np.linspace(0, len(y), 100)
    
    fig, axis = plt.subplots(1, 1, figsize=(6, 4))   
       
    axis.plot(y, "k-", lw=3, label="y", alpha=0.25)
    axis.plot(xf, l5p(xf, *popt), 'k--', lw=1, label="fit")
    
    axis.axvline(x=fC, color="k", linestyle=":", linewidth=1)
    axis.axhline(y=fA, color="k", linestyle=":", linewidth=1)
    axis.axhline(y=fD, color="k", linestyle=":", linewidth=1)

#%%
        
    # Load stack
    # stk = load(stk_paths[stk_idx], crop_size=crop_size)
    
    # # Predict
    # unet = UNet(load_name=model_name)
    # prd = unet.predict(stk, verbose=3)
            
    # # Plot 
    # plt.plot(np.mean(prd, axis=(1, 2)))
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(stk)
    # viewer.add_image(prd)