#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.patch import extract_patches

#%% Inputs --------------------------------------------------------------------

n = 200
patch_size = 256

#%% Initialize ----------------------------------------------------------------

# Paths
dat_path = Path("D:/local_Lebec/data")
trn_path = Path.cwd() / "data" / "train"
img_paths = list(dat_path.rglob("*.tif"))

# Random seed
np.random.seed(42)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    img_idxs = np.random.choice(
        np.arange(len(img_paths)), size=n, replace=False)
    
    for img_idx in img_idxs:

        # Open image
        path = img_paths[img_idx]
        img = io.imread(path)
        
        # Extract patch
        pch = extract_patches(img, patch_size, 0)
        pch_idx = np.random.randint(len(pch))
        
        # Save
        io.imsave(
            trn_path / (path.stem + f"_{pch_idx:02d}.tif"),
            pch[pch_idx], check_contrast=False,                          
            )
