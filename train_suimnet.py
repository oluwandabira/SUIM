"""
# Training pipeline of the SUIM-Net
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
import os
import math
from os.path import join, exists
from keras import callbacks
# local libs
from models.suim_net import SUIM_Net
from utils.data import suim_dataset

import tensorflow as tf
from tensorflow.keras import layers

# dataset directory
dataset_name = "suim"
train_dir = "data/train_val/"

# ckpt directory
ckpt_dir = "ckpt/"
base_ = 'VGG'  # or 'RSB'
if base_ == 'RSB':
    im_res_ = (320, 240, 3)
    ckpt_name = "suimnet_rsb.hdf5"
else:
    im_res_ = (320, 256, 3)
    ckpt_name = "suimnet_vgg.hdf5"
model_ckpt_name = join(ckpt_dir, ckpt_name)
if not exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# initialize model
suimnet = SUIM_Net(base=base_, im_res=im_res_, n_classes=8)
model = suimnet.model
print(model.summary())
# load saved model
#model.load_weights(join("ckpt/saved/", "***.hdf5"))


batch_size = 8
num_epochs = 50
# setup data generator
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

model_checkpoint = callbacks.ModelCheckpoint(model_ckpt_name,
                                             monitor='loss',
                                             verbose=1, mode='auto',
                                             save_weights_only=True,
                                             save_best_only=True)

# data generator
dataset = suim_dataset(train_dir, im_res_[:2])

dataset = dataset.cache().batch(batch_size).prefetch(
    buffer_size=tf.data.AUTOTUNE).repeat()

# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal_and_vertical"),
#     layers.RandomRotation(0.2)
# ])

# fit model
model.fit(dataset,
          steps_per_epoch=5000,
          epochs=num_epochs,
          callbacks=[model_checkpoint])
