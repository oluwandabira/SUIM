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
from keras import layers

from keras.callbacks import ModelCheckpoint, TensorBoard
from models.picker import pick

# dataset directory
train_dir = "data/train_val/"

# ckpt directory
ckpt_dir = "ckpt/saved/"

model, im_res, ckpt_name = pick("deeplabv3")

batch_size = 8
num_epochs = 50
val_split = 0.2


# data generator
dataset = suim_dataset(train_dir, im_res[:2])


dataset = dataset.cache().batch(batch_size).repeat().map(Augment()).prefetch(
    buffer_size=tf.data.AUTOTUNE)


model_checkpoint = ModelCheckpoint(ckpt_name,
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
