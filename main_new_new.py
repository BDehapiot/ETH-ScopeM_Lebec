#%% Imports -------------------------------------------------------------------

import time
import shutil
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
run_process = 1
run_analyse = 0

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
        
    def __init__(self, data_path, crop_size=512):
        
        # Fetch
        self.data_path = data_path
        self.crop_size = crop_size
        
        # run
        self.initialize()
        for self.i, self.path in enumerate(self.paths):
            self.extract()
        
# Method : initialize() -------------------------------------------------------

    def initialize(self):
        
        # Fetch paths & names
        self.paths = [
            f for f in data_path.iterdir() 
            if "TemporaryImages" in str(f)
            and f.is_dir() 
            ]
        self.names  = [path.name for path in self.paths]
        self.mnames = [mapping[name] for name in self.names]
        self.snames = [
            name.replace("TemporaryImages", "") + "_" + mname
            for (name, mname) in zip(self.names, self.mnames)
            ]
        
        # Init outputs directory
        self.out_path = self.data_path / "outputs"  
        if self.out_path.exists():
            for item in self.out_path.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        else:
            self.out_path.mkdir(parents=True, exist_ok=True) 
            
# Method : extract() ----------------------------------------------------------

    def extract(self):
        
        def _extract():
            
        
        t0 = time.time()
        print("extract()    : ", end="", flush=False)
        
        stk = []
        for img_path in list(self.path.glob("*.tif")):
            img = io.imread(img_path)
            if crop_size is not None:
                y0 = (img.shape[0] - self.crop_size) // 2
                x0 = (img.shape[1] - self.crop_size) // 2
                y1, x1 = y0 + self.crop_size, x0 + self.crop_size
                stk.append(img[y0:y1, x0:x1])
            else:
                stk.append(img)
        stk = np.stack(stk)
        
        # Save
        io.imsave(
            self.out_path / (self.snames[self.i] + "_stk.tif"),
            ((self.stk / 4095) * 255).astype("uint8"), 
            check_contrast=False,
            )
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    if run_process:
        process = Process(data_path, crop_size=crop_size)
    names  = process.names
    mnames = process.mnames
    snames = process.snames

    # if run_analyse:
    #     Analyse()
        
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(main.stk)
    # viewer.add_image(main.prd)
