#%% Imports -------------------------------------------------------------------

import time
import pickle
import shutil
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed

# bdtools
from bdtools.norm import norm_pct
from bdtools.models.unet import UNet

# scipy
from scipy.optimize import curve_fit, root_scalar

# matplotlib
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

#%% Inputs --------------------------------------------------------------------

# Procedure
procedure = {
    "extract" : 0,
    "predict" : 0,
    "fit"     : 0,
    "analyse" : 1,
    }

# Parameters
parameters = {
    "data_path"  : Path("D:\local_Lebec\data"),
    "model_name" : "model_256_normal_4000-271_1",
    "sampling"   : 1,
    "crop_size"  : 512,
    }
    
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

#%% Class  : Main -------------------------------------------------------------

class Main:
        
    def __init__(
            self, 
            procedure=procedure,
            parameters=parameters,
            ):
        
        # Fetch
        self.procedure  = procedure
        self.data_path  = parameters["data_path"]
        self.model_name = parameters["model_name"]
        self.sampling   = parameters["sampling"]
        self.crop_size  = parameters["crop_size"] 
        
        # run
        self.initialize()
        if self.procedure["extract"]:
            self.extract()
        if self.procedure["predict"]:
            self.predict()
        if self.procedure["fit"]:
            self.fit()
        if self.procedure["analyse"]:
            self.analyse()
        
#%% Method : initialize() -----------------------------------------------------

    def initialize(self):
        
        # Fetch paths & names
        self.paths = [
            f for f in self.data_path.iterdir() 
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
        if self.procedure["extract"]:
            if self.out_path.exists():
                for item in self.out_path.iterdir():
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            else:
                self.out_path.mkdir(parents=True, exist_ok=True) 
            
#%% Method : extract() --------------------------------------------------------

    def extract(self):
        
        def _extract(i, path):
            
            # Load & crop
            stk = []
            for img_path in list(path.glob("*.tif")):
                img = io.imread(img_path)
                if self.crop_size is not None:
                    y0 = (img.shape[0] - self.crop_size) // 2
                    x0 = (img.shape[1] - self.crop_size) // 2
                    y1, x1 = y0 + self.crop_size, x0 + self.crop_size
                    stk.append(img[y0:y1, x0:x1])
                else:
                    stk.append(img)
            stk = np.stack(stk)
            stk = stk[::self.sampling] # timepoints sampling
            
            # Save
            io.imsave(
                self.out_path / (self.snames[i] + "_stk.tif"),
                ((stk / 4095) * 255).astype("uint8"), 
                check_contrast=False,
                )
            
        t0 = time.time()
        print("extract() : ", end="", flush=False)
        
        Parallel(n_jobs=-1)(
            delayed(_extract)(i, path) 
            for i, path in enumerate(self.paths)
            ) 
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
#%% Method : predict() --------------------------------------------------------

    def predict(self):
        
        t0 = time.time()
        
        # Initialize
        unet = UNet(load_name=self.model_name)
        stk_paths = list(self.data_path.rglob("*_stk.tif"))
    
        # Predict & save (one at a time)
        for i, path in enumerate(stk_paths):
            stk = io.imread(path)
            prd = unet.predict(stk, verbose=1)
            io.imsave(
                self.out_path / (self.snames[i] + "_prd.tif"),
                (prd * 255).astype("uint8"), 
                check_contrast=False,
                )
                
        t1 = time.time()
        print(f"predict : {t1 - t0:.3f}s")    

#%% Method : fit() ------------------------------------------------------------

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
            
            axis.text(t05 / len(y), 0.955, "t05", **text_params)
            axis.text(t05 / len(y), 0.040, f"{fit_data['t05']:.1f}", **text_params)
            axis.text(t50 / len(y), 0.955, "t50", **text_params)
            axis.text(t50 / len(y), 0.040, f"{fit_data['t50']:.1f}", **text_params)
            axis.text(t95 / len(y), 0.955, "t95", **text_params)
            axis.text(t95 / len(y), 0.040, f"{fit_data['t95']:.1f}", **text_params)
            
            # Formatting
            axis.set_title(f"{self.snames[i]}")
            axis.set_ylabel("Avg. prob.")
            axis.set_xlabel("Timepoints")
            axis.legend(loc="center right")
            
            formatter = FuncFormatter(
                lambda x, pos: f'{x * self.sampling:.0f}')
            axis.set_ylim(ymin, ymax)   
            axis.set_xlim(0, len(y)) 
            axis.xaxis.set_major_formatter(formatter)
            
            # Save
            plt.tight_layout()
            plt.savefig(
                self.out_path / (self.snames[i] + "_fit-plot.png"), format="png")
            plt.close(fig)
        
        # Execute -------------------------------------------------------------
        
        t0 = time.time()
        print("fit() : ", end="", flush=False)
        
        # Initialize
        prd_paths = list(self.data_path.rglob("*_prd.tif"))
        
        # Load
        prds = [] 
        for path in prd_paths:
            prds.append(io.imread(path))
        
        # Normalize (0 to 1)
        prds = [norm_pct(prd, pct_low=0, pct_high=100) for prd in prds]
            
        for i, prd in enumerate(prds):
            
            # Fit
            y = np.mean(prd, axis=(1, 2))
            x = np.arange(len(y))
            p0 = [np.min(y), np.max(y), np.median(x), 1.0, 1.0]
            bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(l5p, x, y, p0=p0, bounds=bounds)
        
            # Extract parameters
            fA, fD, fC, fB, fE = popt
            t05 = y2x(fA + (fD - fA) * 0.05)
            t50 = y2x((fD - fA) / 2)
            t95 = y2x(fA + (fD - fA) * 0.95)
            
            fit_data = {
                "name" : self.names[i],
                "dst"  : self.mnames[i].split("-")[0],
                "alg"  : self.mnames[i].split("-")[1],
                "bio"  : self.mnames[i].split("-")[2],
                "rep"  : self.mnames[i].split("-")[3],
                "y"    : y,
                "fA"   : fA, 
                "fD"   : fD, 
                "fE"   : fE,
                "fB"   : fB  / self.sampling, 
                "fC"   : fC  * self.sampling,
                "t05"  : t05 * self.sampling, 
                "t50"  : t50 * self.sampling, 
                "t95"  : t95 * self.sampling,  
                }
            
            # Plot
            plot()
            
            # Save
            pkl_path = self.out_path / (self.snames[i] + "_fit-data.pkl")
            with open(str(pkl_path), "wb") as f:
                pickle.dump(fit_data, f)
        
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")

#%% Method : analyse() --------------------------------------------------------

    def analyse(self):
        
        # Nested function(s) --------------------------------------------------
        
        def format_data():
            
            mrg_data = []
            for i, cond in enumerate(unique_conds):
                t05s, t50s, t95s, ys = [], [], [], []
                for data in fit_data:
                    if data["dst"] == cond[0] and data["alg"] == cond[1]:
                        t05s.append(data["t05"])
                        t50s.append(data["t50"])
                        t95s.append(data["t95"])
                        ys.append(data["y"])
                mrg_data.append({
                    "dst" : cond[0], "alg" : cond[1],
                    "t05_all" : t05s, 
                    "t05_avg" : np.mean(t05s), 
                    "t05_std" : np.std(t05s ),
                    "t50_all" : t50s,
                    "t50_avg" : np.mean(t50s), 
                    "t50_std" : np.std(t50s ),
                    "t95_all" : t95s,
                    "t95_avg" : np.mean(t95s), 
                    "t95_std" : np.std(t95s ),
                    "y_all"   : ys, 
                    "y_avg"   : np.mean(ys, axis=0), 
                    "y_std"   : np.std(ys, axis=0),
                    })
                
            return mrg_data
        
        def plot():
            
            # Initialize
            cmap = plt.get_cmap("viridis", len(unique_conds))
            fig = plt.figure(figsize=(6, 9), layout="tight")
            gs = GridSpec(3, 2, figure=fig)
            
            # Line plots ------------------------------------------------------
            
            ax0 = fig.add_subplot(gs[0, :2])
            for i, cond in enumerate(unique_conds):
                ls = "-" if cond[1] == "para" else  "--"
                color = cmap(i) if cond[1] == "para" else cmap(i - 1)
                ax0.plot(
                    mrg_data[i]["y_avg"], ls, lw=2, color=color, alpha=1.0, 
                    label=f"{cond[0]}, {cond[1]}"
                    )
            ax0.set_title("Avg. prob.")
            ax0.set_ylabel("Avg. prob.")
            ax0.set_xlabel("Timepoints")
            ax0.legend(loc="center right")
            formatter = FuncFormatter(
                lambda x, pos: f"{x * self.sampling:.0f}")
            ax0.xaxis.set_major_formatter(formatter)
            
            # Box plots -------------------------------------------------------
            
            ax1 = fig.add_subplot(gs[1, :2])
            for i, cond in enumerate(unique_conds):
                color = cmap(i) if cond[1] == "para" else cmap(i - 1)
                ax1.boxplot(
                    mrg_data[i]["t50_all"],
                    positions=[i + 1],
                    widths=0.6,
                    showfliers=False,
                    tick_labels=[f"{cond[0]}\n{cond[1]}"],
                    )
                ax1.scatter(
                    np.full(len(mrg_data[i]["t50_all"]), i + 1), 
                    mrg_data[i]["t50_all"], 
                    color=color, 
                    edgecolors="none",
                    alpha=1.0, 
                    s=40, 
                    )
            ax1.set_title("t50")
            ax1.set_ylabel("t50 (timepoints)")
            
            # Scatter plots ---------------------------------------------------
            
            ax2 = fig.add_subplot(gs[2, 0])
            ax3 = fig.add_subplot(gs[2, 1])
            x_vals = [4, 4, 10, 10, 20, 20, 30, 30, 50, 50]
            for i, cond in enumerate(unique_conds):
                color = cmap(i) if cond[1] == "para" else cmap(i - 1)
                if cond[1] == "para":
                    ax2.scatter(
                        np.full(len(mrg_data[i]["t50_all"]), x_vals[i]),
                        mrg_data[i]["t50_all"],
                        color=color,
                        )
                if cond[1] == "perp":
                    ax3.scatter(
                        np.full(len(mrg_data[i]["t50_all"]), x_vals[i]),
                        mrg_data[i]["t50_all"],
                        color=color,
                        )
            ax2.set_title("t50 - para.")
            ax2.set_ylabel("t50 (timepoints)")
            ax2.set_xlabel("Distance (µm)")
            ax2.set_xticks(np.arange(0, 70, 10))
            ax3.set_title("t50 - perp.")
            ax3.set_ylabel("t50 (timepoints)")
            ax3.set_xlabel("Distance (µm)")
            ax3.set_xticks(np.arange(0, 70, 10))
            
            # Save
            plt.tight_layout()
            plt.savefig(
                self.out_path / "0_merged-plot.png", format="png")
            plt.close(fig)
        
        # Execute -------------------------------------------------------------
        
        t0 = time.time()
        print("analyse() : ", end="", flush=False)
        
        # Initialize
        pkl_paths = self.data_path.rglob("*_fit-data.pkl")
               
        # Load
        fit_data = []
        for i, path in enumerate(pkl_paths):
            with open(str(path), "rb") as f:
                fit_data.append(pickle.load(f))

        # Get conditions
        conds = [(data["dst"], data["alg"]) for data in fit_data]
        unique_conds = sorted(list(set(conds)))
        
        # Format data
        mrg_data = format_data()
        
        # Plot
        plot()
        
        # Save
        pkl_path = self.out_path / "0_merged-data.pkl"
        with open(str(pkl_path), "wb") as f:
            pickle.dump(mrg_data, f)
            
        selected_keys = [
            "dst", "alg", 
            "t05_avg", "t05_std",
            "t50_avg", "t50_std",
            "t95_avg", "t95_std",
            ]
        mrg_data = pd.DataFrame(mrg_data)[selected_keys]
        mrg_data.to_csv(self.out_path / "0_merged-data.csv", index=False)
    
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    main = Main()