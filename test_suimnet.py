"""
# Test script for the SUIM-Net
    # for 5 object categories: HD, FV, RO, RI, WR 
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
import numpy as np
# local libs
from models.suim_net import SUIM_Net
from utils.data import suim_dataset

# experiment directories
test_dir = "data/TEST/"


# input/output shapes
base_ = 'VGG'  # or 'RSB'
if base_ == 'RSB':
    im_res_ = (320, 240, 3)
    ckpt_name = "suimnet_rsb.hdf5"
else:
    im_res_ = (320, 256, 3)
    ckpt_name = "suimnet_vgg.hdf5"
suimnet = SUIM_Net(base=base_, im_res=im_res_, n_classes=8)
model = suimnet.model
print(model.summary())
model.load_weights("ckpt/" + ckpt_name)


# test images
dataset = suim_dataset(test_dir, im_res_[:2])

model.evaluate(dataset.batch(8), use_multiprocessing=True)
