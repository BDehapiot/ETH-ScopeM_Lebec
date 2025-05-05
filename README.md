![Python Badge](https://img.shields.io/badge/Python-3.10-rgb(69%2C132%2C182)?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![TensorFlow Badge](https://img.shields.io/badge/TensoFlow-2.10-rgb(255%2C115%2C0)?logo=TensorFlow&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![CUDA Badge](https://img.shields.io/badge/CUDA-11.2-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![cuDNN Badge](https://img.shields.io/badge/cuDNN-8.1-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))    
![Author Badge](https://img.shields.io/badge/Author-Benoit%20Dehapiot-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![Date Badge](https://img.shields.io/badge/Created-2025--03--20-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![License Badge](https://img.shields.io/badge/Licence-GNU%20General%20Public%20License%20v3.0-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))    

# ETH-ScopeM_Lebec  
Deep learning bacteria segmentation (bright-field)

## Index
- [Installation](#installation)
- [Usage](#usage)
- [Comments](#comments)

## Installation

Pease select your operating system

<details> <summary>Windows</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Run the downloaded `.exe` file  
    - Select "Add Miniforge3 to PATH environment variable"  

### Step 3: Setup Conda 
- Open the newly installed Miniforge Prompt  
- Move to the downloaded GitHub repository
- Run one of the following command:  
```bash
# TensorFlow with GPU support
mamba env create -f environment_tf-gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment_tf-nogpu.yml
```  
- Activate Conda environment:
```bash
conda activate lebec
```
Your prompt should now start with `(lebec)` instead of `(base)`

</details> 

<details> <summary>MacOS</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Open your terminal
- Move to the directory containing the Miniforge installer
- Run one of the following command:  
```bash
# Intel-Series
bash Miniforge3-MacOSX-x86_64.sh
# M-Series
bash Miniforge3-MacOSX-arm64.sh
```   

### Step 3: Setup Conda 
- Re-open your terminal 
- Move to the downloaded GitHub repository
- Run one of the following command: 
```bash
# TensorFlow with GPU support
mamba env create -f environment_tf-gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment_tf-nogpu.yml
```  
- Activate Conda environment:  
```bash
conda activate lebec
```
Your prompt should now start with `(lebec)` instead of `(base)`

</details>


## Usage

<p align="left">
  <img src="utils/example_stk.jpg" alt="example_stk" width="192" />
  <img src="utils/example_prd.jpg" alt="example_prd" width="192" />
  <img src="utils/example_fit-plot.jpg" alt="example_fit-plot" height="192" />
</p>

### `main.py`

#### Procedure

- **extract()**  
Extract and save cropped movies as `..._stk.tif` in the `outputs` 
directory.

- **predict()**  
Batch read `..._stk.tif`, predict segmentation masks.  
Predictions are saves as `..._prd.tif` in the `outputs` directory.

- **fit()**  
Measure avg. predictions (probabilities) over time and fit using L5P.  
Fit data are saved as `..._fit-data.pkl` in the `outputs` directory.  
Fit plots are saved as `..._fit-plot.png` in the `outputs` directory.  
L5P (Five Parameters logistic regression)

- **analyse()**  
Merge and average fit data per condition (distance & alignement).  
Merge data are saved as `..._merged-data.pkl/csv` in the `outputs` directory.   
Merge plots are saved as `..._merged-plot.png` in the `outputs` directory. 

#### Parameters
```bash
- data_path  # str, path to the data directory
- model_name # str, model name for predictions (model should be in root)
- sampling   # int, temporal sampling (1 = all frames, n = every [n]th frames)
- crop_size  # int, size in pixels of the central cropped region
```

#### Outputs
```bash
# One per movie:
- ..._stk.tif       # uint8, cropped movie
- ..._prd.tif       # uint8, cropped movie predictions
- ..._fit-data.pkl  # fit data (read with pickle Python module)
- ..._fit-plot.png  # fit data plot overview

# One for all:
- 0_merged-data.pkl # merged data (read with pickle Python module)
- 0_merged-data.csv # selected data saved as CSV
    - dst           # bacteria laying distance
    - alg           # bacteria laying alignement
    - t05_avg/std   # time of 05% of max avg. prob. (from fit)
    - t50_avg/std   # time of 50% of max avg. prob. (from fit)
    - t95_avg/std   # time of 95% of max avg. prob. (from fit)
- 0_merged-plot.png # merged data plot overview
```

### `extract.py`
Extract random patches from dataset.  

### `train.py`
Annotate extracted patches & train DL model (U-Net with resnet18 encoder). 


## Comments