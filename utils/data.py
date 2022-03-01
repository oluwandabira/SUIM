from typing import List, Tuple, Generator
from pathlib import Path
from cv2 import flip

import numpy as np
import tensorflow as tf
from keras import layers
import cv2

categories_code = [
    "BW", "HD", "PF", "WR",
    "RO", "RI", "FV", "SR"
]


def split_categories(mask: np.ndarray) -> np.ndarray:
    imw, imh = mask.shape[0], mask.shape[1]

    categories = np.zeros((imw, imh, 8))

    for i in range(imw):
        for j in range(imh):
            code = int("".join([str(i) for i in mask[i, j].tolist()]), 2)
            categories[i, j, code] = 1

    return categories


rgbarray = np.array([
    0, 0, 0,
    0, 0, 1,
    0, 1, 0,
    0, 1, 1,
    1, 0, 0,
    1, 0, 1,
    1, 1, 0,
    1, 1, 1
]).reshape((8, 3))


def join_categories(masks: np.ndarray) -> np.ndarray:
    imw, imh = masks.shape[0], masks.shape[1]

    mask = np.zeros((imw, imh, 3), dtype=np.uint8)

    for i in range(imw):
        for j in range(imh):
            # for c in range(masks.shape[2]):
            #     if masks[i, j, c]:
            #         mask[i, j] = rgbarray[c]
            #         break
            amax = masks[i, j].argmax()
            idx = amax if amax.shape == () else amax[0]
            mask[i, j] = rgbarray[idx]

    return mask


def list_categories(categories: np.ndarray) -> List[str]:
    return [cat for code, cat in enumerate(categories_code) if (np.any(categories[:, :, code]))]


def generate_images_masks(data_dir: str, img_size: Tuple[int, int]) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    path = Path(data_dir)
    images = sorted([f for f in path.glob("images/*") if f.is_file()])
    masks = sorted([f for f in path.glob("masks/*") if f.is_file()])

    assert len(images) == len(
        masks), "Unequal number of detected images and masks"

    for img_file, mask_file in zip(images, masks):
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255

        mask = cv2.imread(str(mask_file))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, img_size)

        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        yield img, split_categories(np.uint8(mask))


def suim_dataset(data_dir, img_size):
    img_gen = generate_images_masks(data_dir, img_size)
    return tf.data.Dataset.from_generator(
        lambda: (x for x in img_gen),
        output_signature=(
            tf.TensorSpec(
                shape=(img_size[1], img_size[0], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(img_size[1], img_size[0], 8), dtype=tf.uint8)
        ))


def augment(flip_mode, rotation_range, seed=123):
    return tf.keras.Sequential([
        layers.RandomFlip(flip_mode, seed=seed),
        layers.RandomRotation(rotation_range, seed=seed),
    ])


class Augment(layers.Layer):
    def __init__(self, flip_mode="horizontal_and_vertical", rotation_range=0.02, seed=123, **kwargs):
        super().__init__(**kwargs)
        self.img_aug = augment(flip_mode, rotation_range, seed)
        self.mask_aug = augment(flip_mode, rotation_range, seed)

    def call(self, img, img_mask):
        return self.img_aug(img), self.mask_aug(img_mask)
