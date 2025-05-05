#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt 

# bdtools
from bdtools.models.unet import UNet

# scipy
from scipy.optimize import curve_fit, root_scalar

#%% Inputs --------------------------------------------------------------------

# Procedure
run_process = 0
run_analyse = 1

# Parameters
crop_size = 256

#%% Paths ---------------------------------------------------------------------

data_path = Path("D:\local_Lebec\data")
model_name = "model_256_normal_4000-271_1"
stk_paths = [
    f for f in data_path.iterdir() 
    if "TemporaryImages" in str(f)
    and f.is_dir() 
    ]

#%% Mapping -------------------------------------------------------------------

mapping = {
    
    "CAP047_XY13TemporaryImages" : "04um-para-bio1-rep1",
    "CAP047_XY14TemporaryImages" : "04um-para-bio1-rep2",
    "CAP047_XY09TemporaryImages" : "04um-perp-bio1-rep1",
    "CAP047_XY10TemporaryImages" : "04um-perp-bio1-rep2",
    "CAP047_XY05TemporaryImages" : "10um-para-bio1-rep1",
    "CAP047_XY06TemporaryImages" : "10um-para-bio1-rep2",
    "CAP047_XY01TemporaryImages" : "10um-perp-bio1-rep1",
    "CAP047_XY02TemporaryImages" : "10um-perp-bio1-rep2",
    "CAP045_XY15TemporaryImages" : "20um-para-bio1-rep1",
    "CAP045_XY16TemporaryImages" : "20um-para-bio1-rep2",
    "CAP045_XY11TemporaryImages" : "20um-perp-bio1-rep1",
    "CAP045_XY12TemporaryImages" : "20um-perp-bio1-rep2",
    "CAP045_XY21TemporaryImages" : "30um-para-bio1-rep1",
    "CAP045_XY22TemporaryImages" : "30um-para-bio1-rep2",
    "CAP045_XY18TemporaryImages" : "30um-perp-bio1-rep1",
    "CAP045_XY19TemporaryImages" : "30um-perp-bio1-rep2",
    "CAP056_XY07TemporaryImages" : "50um-para-bio1-rep1",
    "CAP056_XY08TemporaryImages" : "50um-para-bio1-rep2",
    "CAP056_XY02TemporaryImages" : "50um-perp-bio1-rep1",
    "CAP056_XY04TemporaryImages" : "50um-perp-bio1-rep2",
    
    "CAP050_XY13TemporaryImages" : "04um-para-bio2-rep1",
    "CAP050_XY14TemporaryImages" : "04um-para-bio2-rep2",
    "CAP050_XY10TemporaryImages" : "04um-perp-bio2-rep1",
    "CAP050_XY11TemporaryImages" : "04um-perp-bio2-rep2",
    "CAP050_XY07TemporaryImages" : "10um-para-bio2-rep1",
    "CAP050_XY08TemporaryImages" : "10um-para-bio2-rep2",
    "CAP050_XY01TemporaryImages" : "10um-perp-bio2-rep1",
    "CAP050_XY02TemporaryImages" : "10um-perp-bio2-rep2",
    "CAP047_XY29TemporaryImages" : "20um-para-bio2-rep1",
    "CAP047_XY30TemporaryImages" : "20um-para-bio2-rep2",
    "CAP047_XY25TemporaryImages" : "20um-perp-bio2-rep1",
    "CAP047_XY26TemporaryImages" : "20um-perp-bio2-rep2",
    "CAP047_XY49TemporaryImages" : "30um-para-bio2-rep1",
    "CAP047_XY50TemporaryImages" : "30um-para-bio2-rep2",
    "CAP047_XY53TemporaryImages" : "30um-perp-bio2-rep1",
    "CAP047_XY54TemporaryImages" : "30um-perp-bio2-rep2",
    "CAP057_XY05TemporaryImages" : "50um-para-bio2-rep1",
    "CAP057_XY06TemporaryImages" : "50um-para-bio2-rep2",
    "CAP057_XY02TemporaryImages" : "50um-perp-bio2-rep1",
    "CAP057_XY04TemporaryImages" : "50um-perp-bio2-rep2",

    "CAP053_XY14TemporaryImages" : "04um-para-bio3-rep1",
    "CAP053_XY15TemporaryImages" : "04um-para-bio3-rep2",
    "CAP049_XY01TemporaryImages" : "04um-perp-bio3-rep1",
    "CAP049_XY02TemporaryImages" : "04um-perp-bio3-rep2",
    "CAP048_XY33TemporaryImages" : "10um-para-bio3-rep1",
    "CAP048_XY34TemporaryImages" : "10um-para-bio3-rep2",
    "CAP053_XY37TemporaryImages" : "10um-perp-bio3-rep1",
    "CAP053_XY38TemporaryImages" : "10um-perp-bio3-rep2",
    "CAP050_XY21TemporaryImages" : "20um-para-bio3-rep1",
    "CAP050_XY22TemporaryImages" : "20um-para-bio3-rep2",
    "CAP050_XY17TemporaryImages" : "20um-perp-bio3-rep1",
    "CAP050_XY18TemporaryImages" : "20um-perp-bio3-rep2",
    "CAP050_XY30TemporaryImages" : "30um-para-bio3-rep1",
    "CAP050_XY31TemporaryImages" : "30um-para-bio3-rep2",
    "CAP050_XY25TemporaryImages" : "30um-perp-bio3-rep2",
    "CAP050_XY26TemporaryImages" : "30um-perp-bio3-rep2",
    "CAP058_XY05TemporaryImages" : "50um-para-bio3-rep1",
    "CAP058_XY06TemporaryImages" : "50um-para-bio3-rep2",
    "CAP058_XY01TemporaryImages" : "50um-perp-bio3-rep1",
    "CAP058_XY02TemporaryImages" : "50um-perp-bio3-rep2",

    }

#%% Class : Process() ---------------------------------------------------------

class Process:
        
    def __init__(self, path, crop_size=512):
        
        print(f"\n{path.stem}")
        
        # Fetch
        self.path = path
        self.crop_size = crop_size
        
        # Initialize
        self.name = self.path.stem.replace("TemporaryImages", "")
        self.out_path = self.path.parent / "outputs"  
        if not self.out_path.exists():
            self.out_path.mkdir(parents=True, exist_ok=True)            
        
        # Procedure
        self.load()
        self.predict()
        self.fit()
        self.save()

# Method : load() -------------------------------------------------------------

    def load(self):
        
        t0 = time.time()
        print("load()    : ", end="", flush=False)
        
        self.stk = []
        for img_path in list(self.path.glob("*.tif")):
            img = io.imread(img_path)
            if crop_size is not None:
                y0 = (img.shape[0] - self.crop_size) // 2
                x0 = (img.shape[1] - self.crop_size) // 2
                y1, x1 = y0 + self.crop_size, x0 + self.crop_size
                self.stk.append(img[y0:y1, x0:x1])
            else:
                self.stk.append(img)
        self.stk = np.stack(self.stk)
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
# Method : predict() ----------------------------------------------------------
        
    def predict(self):
        
        t0 = time.time()
        print("predict() : ", end="", flush=False)
        
        unet = UNet(load_name=model_name)
        self.prd = unet.predict(self.stk, verbose=0)
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
# Method : fit() --------------------------------------------------------------       

    def fit(self):
        
        # Nested function(s) --------------------------------------------------
        
        def l5p(x, A, D, C, B, E):
            
            """
            A = lower asymptote
            D = upper asymptote
            C = inflection point
            B = slope factor
            E = asymmetry factor (E=1 reduces to 4PL)
            
            """
            return A + (D - A) / ((1 + np.exp(-B*(x - C)))**E)
        
        def y2x(y_target):
            def func(x):
                return l5p(x, *popt) - y_target
            sol = root_scalar(func, bracket=(0, len(y)), method='brentq')
            return sol.root
        
        def plot():
                            
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
            axis.set_title(f"{self.name}")
            axis.set_ylabel("Avg. prob.")
            axis.set_xlabel("Time (timepoints)")
            axis.set_ylim(ymin, ymax)   
            axis.set_xlim(0, len(y))         
            axis.legend(loc="center right")
            
            # Save
            plt.tight_layout()
            plt.savefig(self.out_path / (self.name + "_plot.png"), format="png")
            plt.close(fig)
        
        # Execute -------------------------------------------------------------
        
        t0 = time.time()
        print("fit()     : ", end="", flush=False)
        
        # Fit
        y = np.mean(self.prd, axis=(1, 2))
        x = np.arange(len(y))
        p0 = [np.min(y), np.max(y), np.median(x), 1.0, 1.0]
        bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
        popt, pcov = curve_fit(l5p, x, y, p0=p0, bounds=bounds)
        
        # Extract parameters
        fA, fD, fC, fB, fE = popt
        t05 = y2x(fA + (fD - fA) * 0.05)
        t50 = y2x((fD - fA) / 2)
        t95 = y2x(fA + (fD - fA) * 0.95)
        
        self.fdata = pd.DataFrame([{
            # "popt" : popt, "pcov" : pcov,
            "fA" : fA, "fD" : fD, "fC" : fC, "fB" : fB, "fE" : fE,
            "t05" : t05, "t50" : t50, "t95" : t95,            
            }])
        
        # Plot
        plot()

        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
                        
# Method : save() -------------------------------------------------------------            
        
    def save(self):
        
        t0 = time.time()
        print("save()    : ", end="", flush=False)
                
        self.fdata.to_csv(
            self.out_path / (self.name + "_fdata.csv"), index=False
            )
        
        io.imsave(
            self.out_path / (self.name + "_stk.tif"),
            ((self.stk / 4095) * 255).astype("uint8"), 
            check_contrast=False,
            )
        
        io.imsave(
            self.out_path / (self.name + "_prd.tif"),
            (self.prd * 255).astype("uint8"), 
            check_contrast=False,
            )
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Class : Analyse() ---------------------------------------------------------
        
class Analyse:
    
    def __init__(self):
        
        # Fetch
        # Initialize
        # Procedure
        
        pass

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    if run_process:
        for stk_path in stk_paths:
            Process(stk_path, crop_size=crop_size)
            
    if run_analyse:
        Analyse()
        
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(main.stk)
    # viewer.add_image(main.prd)
