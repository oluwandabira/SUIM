from typing import List, Tuple
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2

categories_code = [
    "BW", "HD", "PF", "WR",
    "RO", "RI", "FV", "SR"
]


def get_categories(mask: np.ndarray) -> np.ndarray:
    imw, imh = mask.shape[0], mask.shape[1]

    categories = np.zeros((imw, imh, 8))

    for i in range(imw):
        for j in range(imh):
            code = int("".join([str(i) for i in mask[i, j].tolist()]), 2)
            categories[i, j, code] = 1

    return categories


def list_categories(categories: np.ndarray) -> List[str]:
    return [cat for code, cat in enumerate(categories_code) if (np.any(categories[:, :, code]))]


def generate_images_masks(data_dir: str, img_size: Tuple[int, int]):
    path = Path(data_dir)
    images = sorted([f for f in path.glob("images/*") if f.is_file()])
    masks = sorted([f for f in path.glob("masks/*") if f.is_file()])

    for img_file, mask_file in zip(images, masks):
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255

        mask = cv2.imread(str(mask_file))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, img_size)

        mask = mask / 255
        yield img, get_categories(np.uint8(mask))


def suim_dataset(data_dir, img_size):
    img_gen = generate_images_masks(data_dir, img_size)
    return tf.data.Dataset.from_generator(
        lambda: (x for x in img_gen),
        output_signature=(
            tf.TensorSpec(
                shape=(img_size[1], img_size[0], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(img_size[1], img_size[0], 8), dtype=tf.uint8)
        ))
