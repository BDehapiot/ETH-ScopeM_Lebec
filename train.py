#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools.models.annotate import Annotate
from bdtools.models.unet import UNet

#%% Inputs --------------------------------------------------------------------

# Procedure
annotate = 0
train = 0
predict = 1

# UNet build()
backbone = "resnet18"
activation = "sigmoid"
downscale_factor = 1

# UNet train()
preview = 0

# preprocess
patch_size = 256
patch_overlap = 0
img_norm = "image"
msk_type = "normal"

# augment
iterations = 2000
gamma_p = 0.5
gblur_p = 0.5
noise_p = 0.5 
flip_p = 0.5 
distord_p = 0.5

# train
epochs = 100
batch_size = 8
validation_split = 0.2
metric = "soft_dice_coef"
learning_rate = 0.0005
patience = 20

# predict
stk_idx = 32
model_name = "model_256_normal_2000-55_1"

#%% Initialize ----------------------------------------------------------------

dat_path = Path("D:\local_Lebec\data")
trn_path = Path("data", "train")
stk_paths = [f for f in dat_path.iterdir() if f.is_dir()]

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    if annotate:
        Annotate(trn_path)
    
    if train:
    
        # Load data
        imgs, msks = [], []
        for path in list(trn_path.glob("*.tif")):
            if "mask" in path.name:
                msks.append(io.imread(path))   
                imgs.append(io.imread(str(path).replace("_mask", "")))
        imgs = np.stack(imgs)
        msks = np.stack(msks)
         
        unet = UNet(
            save_name="",
            load_name="",
            root_path=Path.cwd(),
            backbone=backbone,
            classes=1,
            activation=activation,
            )
        
        unet.train(
            
            imgs, msks, 
            X_val=None, y_val=None,
            preview=preview,
            
            # Preprocess
            img_norm=img_norm, 
            msk_type=msk_type, 
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            downscaling_factor=downscale_factor, 
            
            # Augment
            iterations=iterations,
            gamma_p=gamma_p, 
            gblur_p=gblur_p, 
            noise_p=noise_p, 
            flip_p=flip_p, 
            distord_p=distord_p,
            
            # Train
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            metric=metric,
            learning_rate=learning_rate,
            patience=patience,
            
            )
        
    if predict:
        
        def load_stack(stk_path):
            img_paths = list(stk_path.glob("*.tif"))
            stk = []
            for img_path in img_paths:
                stk.append(io.imread(img_path))
            return np.stack(stk)
        
        # Load stack
        stk = load_stack(stk_paths[stk_idx])[:50, 512:1536, 512:1536]
        
        # Predict
        unet = UNet(load_name=model_name)
        prd = unet.predict(stk, verbose=3)
                
        # Display
        viewer = napari.Viewer()
        viewer.add_image(stk)
        viewer.add_image(prd)