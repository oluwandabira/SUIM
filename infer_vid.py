
"""
"""
from ast import arg
import os
import numpy as np
import cv2
import tensorflow as tf
import time
import argparse
from os.path import join, exists
# local libs
from models.suim_net import SUIM_Net
from utils.data import categories_code, join_categories

parser = argparse.ArgumentParser("Run inference on a video")

parser.add_argument("input")
parser.add_argument("out")
parser.add_argument("--threshold", type=float, default=0.5)

args = parser.parse_args()

thres = args.threshold

# input/output shapes
base_ = 'VGG'  # or 'RSB'

im_res_ = (320, 256, 3)
ckpt_name = "suimnet_vgg.hdf5"
suimnet = SUIM_Net(base=base_, im_res=im_res_, n_classes=8)
model = suimnet.model
# print(model.summary())
model.load_weights(join("ckpt/", ckpt_name))

im_h, im_w = im_res_[1], im_res_[0]

#vid = cv2.VideoCapture("vid.AVI")
vid = cv2.VideoCapture(args.input)

frame_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Video writer issues: https://github.com/ContinuumIO/anaconda-issues/issues/223
# output = cv2.VideoWriter("vid_out_.avi", cv2.VideoWriter_fourcc(
#    *'MJPG'), cv2.CAP_PROP_FPS, frame_size)


out_folder = args.out
#out_folder = f"output{time.time_ns()}"
os.mkdir(out_folder)
outputs = [cv2.VideoWriter(f"{out_folder}/{i}.avi", cv2.VideoWriter_fourcc(
    *'MJPG'), vid.get(cv2.CAP_PROP_FPS), frame_size, False) for i in range(8)]


# for i in range(8):
#     os.mkdir(f"{out_folder}/{i}")


tic_tocs = np.zeros(int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))

count = 0
while vid.isOpened():
    ret, orig = vid.read()
    if not ret:
        break
    frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (im_w, im_h))
    frame = np.expand_dims(frame / 255, 0)
    tic = time.perf_counter_ns()
    predicted = model(frame)
    toc = time.perf_counter_ns()
    tic_tocs[count] = toc - tic
    img = np.squeeze(predicted).copy()
    img[img > thres] = 1
    img[img <= thres] = 0
    resized = cv2.resize(img * 255, frame_size)
    for i in range(8):
        #cv2.imwrite(f"{out_folder}/{i}/{count}.jpg", resized[:, :, i])
        #outputs[i].write(cv2.cvtColor(resized[:, :, i], cv2.COLOR_GRAY2BGR))
        outputs[i].write(np.uint8(resized[:, :, i]))
    #img = join_categories(img)
    # output.write(resized)
    count = count + 1
    # cv2.imshow("Image", orig)
    # cv2.imshow("Prediction", cv2.resize(img * 255, frame_size))
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

# cv2.destroyAllWindows()

avg_time = np.average(tic_tocs) * 1e9
print(f"Average frame time is {avg_time}s ({avg_time / 60} fps)")
vid.release()

for i in range(8):
    outputs[i].release()
# dataset = video.video_dataset("vid.AVI", (im_w, im_h))

# predicted = model.predict(dataset.batch(
#     8).prefetch(buffer_size=tf.data.AUTOTUNE))
# print(predicted.shape)
