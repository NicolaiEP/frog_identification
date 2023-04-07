import os
import argparse
import random
from shutil import copy

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def get_data_pairs(data_dir):
    """Get pairs of image and annotation paths from the data directory"""
    image_dir = os.path.join(data_dir, "frog_identification\data\images")
    ann_dir = os.path.join(data_dir, "frog_identification\data\lables")
    pairs = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            ann_path = os.path.join(ann_dir, filename.replace(".jpg", ".txt"))
            if os.path.exists(ann_path):
                pairs.append((image_path, ann_path))
    return pairs


def split_data(pairs, train_dir, val_dir, val_size=0.2):
    """Split data into training and validation sets"""
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    train_pairs, val_pairs = train_test_split(pairs, test_size=val_size)
    for src_image_path, src_ann_path in train_pairs:
        dst_image_path = os.path.join(train_dir, os.path.basename(src_image_path))
        dst_ann_path = os.path.join(train_dir, os.path.basename(src_ann_path))
        copy(src_image_path, dst_image_path)
        copy(src_ann_path, dst_ann_path)
    for src_image_path, src_ann_path in val_pairs:
        dst_image_path = os.path.join(val_dir, os.path.basename(src_image_path))
        dst_ann_path = os.path.join(val_dir, os.path.basename(src_ann_path))
        copy(src_image_path, dst_image_path)
        copy(src_ann_path, dst_ann_path)


def get_model(input_shape):
    """Create and compile model"""
    base_model = models.Sequential()
    base_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    base_model.add(layers.MaxPooling2D((2, 2)))
    base_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
