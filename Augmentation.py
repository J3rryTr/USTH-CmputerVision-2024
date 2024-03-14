import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch

root = f"datasets/"
annotations_folder = f"datasets/wider_face_split/"
train_bbx_path = f"{annotations_folder}wider_face_train_bbx_gt.txt"
val_bbx_path = f"{annotations_folder}wider_face_val_bbx_gt.txt"


def load_bbx(bbx_path):
    with open(bbx_path, mode='r') as file:
        lines = file.readlines()

    annotations = {}
    i = 0
    while i < len(lines):
        file_name = lines[i].strip()
        i += 1
        try:
            num_boxes = int(lines[i].strip())
        except ValueError:
            continue
        i += 1
        boxes = []
        for _ in range(num_boxes):
            box_info = lines[i].strip().split()
            box = {
                'x': int(box_info[0]),
                'y': int(box_info[1]),
                'w': int(box_info[2]),
                'h': int(box_info[3]),
            }
            boxes.append(box)
            i += 1
        annotations[file_name] = boxes

    return annotations


def annotation_to_df(annotation, img_shape):
    cs, xs, ys, ws, hs = [], [], [], [], []

    h, w, _ = img_shape
    for box in annotation:
        cs.append(0)
        xs.append((box["x"] + box["w"] / 2.0) / w)
        ys.append((box["y"] + box["h"] / 2.0) / h)
        ws.append(box["w"] / w)
        hs.append(box["h"] / h)

    return pd.DataFrame({0: cs, 1: xs, 2: ys, 3: ws, 4: hs})


train_annotations = load_bbx(train_bbx_path)
val_annotations = load_bbx(val_bbx_path)

train_keys = []
val_keys = []
for key in train_annotations.keys():
    train_keys.append(key)
for key in val_annotations.keys():
    val_keys.append(key)


def plot_boxes(img, df):
    h, w, _ = img.shape

    fig, ax = plt.subplots()
    for index, row in df.iterrows():
        patch = Rectangle(
            ((row[1] - row[3] / 2.0) * w, (row[2] - row[4] / 2.0) * h),
            row[3] * w,
            row[4] * h,
            edgecolor='red',
            fill=False,
        )
        ax.add_patch(patch)

    plt.imshow(img)


def add_dataset(keys, annotations, img_folder, root, split):
if not os.path.exists(root):
    os.makedirs(root)
if not os.path.exists(f"{root}images/{split}"):
    os.makedirs(f"{root}images/{split}")
    os.makedirs(f"{root}labels/{split}")

for i, key in enumerate(keys):
    img = np.array(Image.open(f"{img_folder}{key}"))
    Image.fromarray(img).save(f"{root}images/{split}/im{i}.jpg")
    df = annotation_to_df(annotations[key], img.shape)
    df.to_csv(f"{root}labels/{split}/im{i}.txt", header=False, index=False, sep='\t')