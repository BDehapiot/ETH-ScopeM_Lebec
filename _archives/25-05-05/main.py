#%% Imports -------------------------------------------------------------------

import shutil
import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt 

# bdtools
from bdtools.models.unet import UNet

# scipy
from scipy.optimize import curve_fit, root_scalar

#%% Inputs --------------------------------------------------------------------

# Parameters
crop_size = 512

# predict
stk_idx = 28 # ok(0, 5), bof(25)

#%% Initialize ----------------------------------------------------------------

dat_path = Path("D:\local_Lebec\data")
model_name = "model_256_normal_4000-271_1"
trn_path = Path("data", "train")
stk_paths = [
    f for f in dat_path.iterdir() 
    if "TemporaryImages" in str(f)
    and f.is_dir() 
    ]

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

#%% Function : l5p_fit() ------------------------------------------------------

def l5p_fit(y, outputs_path, plot=True, display=False, save=True):

    # Nested function(s) ------------------------------------------------------

    def l5p(x, A, D, C, B, E):
        
        """
        A = lower asymptote
        D = upper asymptote
        C = inflection point (EC50)
        B = slope factor
        E = asymmetry factor (E=1 reduces to 4PL)
        
        """
        return A + (D - A) / ((1 + np.exp(-B*(x - C)))**E)
    
    def y2x(y_target, popt, x_bounds=(0, 100)):
        def func(x):
            return l5p(x, *popt) - y_target
        sol = root_scalar(func, bracket=x_bounds, method='brentq')
        return sol.root
    
    def plot(fit_data, display=False, save=True):
        
        # Fetch
        y = fit_data["y"]
        popt = fit_data["popt"]
        fA, fD, fC, fB, fE = fit_data["params"]
        t05, t50, t95 = fit_data["tmarks"]
                
        # Initialize
        xf = np.linspace(0, len(y), 100)
        
        # Main plot
        fig, axis = plt.subplots(1, 1, figsize=(6, 4))   
        axis.plot(y, "k-", lw=5, label="y", alpha=0.25)
        axis.plot(xf, l5p(xf, *popt), 'r-', lw=1, label="fit")
            
        # Markers
        ymin, ymax = -0.1, 1.1
        fApc = fA / (ymax - ymin) + np.abs(ymin - fA) / (ymax - ymin)
        fDpc = fD / (ymax - ymin) + np.abs(ymin - fA) / (ymax - ymin)
        
        axis.axvline(
            x=t05, ymin=fApc, ymax=fDpc, 
            color="k", linestyle="--", linewidth=0.5
            )
        axis.axvline(
            x=t50, ymin=fApc, ymax=fDpc, 
            color="k", linestyle="-" , linewidth=1
            )
        axis.axvline(
            x=t95, ymin=fApc, ymax=fDpc, 
            color="k", linestyle="--", linewidth=0.5
            )
        axis.axhline(y=fA , color="k", linestyle="--", linewidth=0.5)
        axis.axhline(y=fD , color="k", linestyle="--", linewidth=0.5)
        
        text_params = {
            "size" : 10, "color" : "k", 
            "transform": axis.transAxes, "ha": "center", "va": "center"
            }
        
        axis.text(t05 / len(y), 0.955, "t05",        **text_params)
        axis.text(t05 / len(y), 0.040, f"{t05:.1f}", **text_params)
        axis.text(t50 / len(y), 0.955, "t50",        **text_params)
        axis.text(t50 / len(y), 0.040, f"{t50:.1f}", **text_params)
        axis.text(t95 / len(y), 0.955, "t95",        **text_params)
        axis.text(t95 / len(y), 0.040, f"{t95:.1f}", **text_params)
        
        # Formatting
        axis.set_title(f"{stk_path.name}")
        axis.set_ylabel("Avg. prob.")
        axis.set_xlabel("Time (timepoints)")
        axis.set_ylim(ymin, ymax)   
        axis.set_xlim(0, len(y))         
        axis.legend(loc="center right")
        
        # # Save
        # plt.tight_layout()
        # plt.savefig(outputs_path / (plot_stem + ".png"), format="png")
        # plt.close(fig)
        # # plt.show()
        
        return fig

    # Execute -----------------------------------------------------------------    

    # Fit
    x = np.arange(len(y))
    p0 = [np.min(y), np.max(y), np.median(x), 1.0, 1.0]
    bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
    popt, pcov = curve_fit(l5p, x, y, p0=p0, bounds=bounds)
    
    # Extract
    fA, fD, fC, fB, fE = popt
    t05 = y2x(fA + (fD - fA) * 0.05, popt, x_bounds=(0, len(y)))
    t50 = y2x((fD - fA) / 2, popt, x_bounds=(0, len(y)))
    t95 = y2x(fA + (fD - fA) * 0.95, popt, x_bounds=(0, len(y)))
    fit_data = {
        "y": y, "x": x, 
        "popt": popt, "pcov": pcov,
        "params": (fA, fD, fC, fB, fE),
        "tmarks": (t05, t50, t95),
        }
    
    # Plot
    fit_plot = plot(fit_data, display=display, save=save)
    
    return fit_data, fit_plot

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
            
    # for stk_path in stk_paths:
    stk_path = stk_paths[stk_idx]
    print(stk_path)
        
    # Initialize
    stk_name = stk_path.stem.replace("TemporaryImages", "")
    outputs_path = stk_path.parent / "outputs"  
    if outputs_path.exists():
        for item in outputs_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        outputs_path.mkdir(parents=True, exist_ok=True)
        
    # # Load
    # stk = load(stk_path, crop_size=crop_size)
    
    # # Predict
    # unet = UNet(load_name=model_name)
    # prd = unet.predict(stk, verbose=3)
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(stk)
    # viewer.add_image(prd)
    
#%%
    
    # # Fit
    # y = np.mean(prd, axis=(1, 2))
    # fit_data, fit_plot = l5p_fit(y)
    # fit_plot.show()

    # # Save
    # io.imsave(
    #     output_path / (output_name + "_stk.tif"),
    #     stk, check_contrast=False
    #     )
    # io.imsave(
    #     output_path / (output_name + "_prd.tif"),
    #     prd.astype("float32"), check_contrast=False
    #     )  
    