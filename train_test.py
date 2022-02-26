"""
# Training pipeline of the SUIM-Net
# Paper: https://arxiv.org/pdf/2004.01241.pdf  
"""
import os
import datetime
from os.path import join, exists
# local libs
from models.suim_net import SUIM_Net
from utils.data import suim_dataset, Augment

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorboard

# dataset directory
dataset_name = "suim"
train_dir = "data/train_val/"
test_dir = "data/TEST/"

# ckpt directory
ckpt_dir = "ckpt/saved/"
base_ = 'RSB'  # or 'VGG'
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

batch_size = 8
num_epochs = 50
val_split = 0.2


# data generator
dataset = suim_dataset(train_dir, im_res_[:2])


dataset = dataset.cache().batch(batch_size).repeat().map(Augment()).prefetch(
    buffer_size=tf.data.AUTOTUNE)


model_checkpoint = ModelCheckpoint(model_ckpt_name,
                                   monitor='loss',
                                   verbose=1, mode='auto',
                                   save_weights_only=True,
                                   save_best_only=True)

log_dir = "data/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# fit model
model.fit(dataset,
          steps_per_epoch=5000,
          epochs=num_epochs,
          verbose=2,
          callbacks=[model_checkpoint, tensorboard_callback],
          )


# test images
test_dataset = suim_dataset(test_dir, im_res_[:2])

model.evaluate(dataset.cache().batch(8).prefetch(
    buffer_size=tf.data.AUTOTUNE))
