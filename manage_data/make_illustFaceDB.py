# %%
import csv
import os
import shutil
import glob
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
# %%


class MyRotationTransform:
    def __init__(self, angle):
        self.angles = angle

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


class MyRandomCrop:
    def __init__(self):
        pass

    def __call__(self, img, label, face_flag=True):
        _, x, y, size = tuple(map(int, label))
        h, w = np.array(img).shape[:2]
        if face_flag:
            size2 = min([x, y, h-y, w-x])
            new_size = random.choice(list(range(max(size2-size, 1))))+size
            img = TF.crop(img, y-new_size, x-new_size, 2*new_size, 2*new_size)
            img = transforms.Resize(IMAGE_SIZE)(img)
        else:
            list_ = [x-size, y-size, h-(y+size), w-(x+size)]
            if h-(y+size) == max(list_):
                img = TF.crop(img, y+size, 0, h-(y+size), w)
            elif w-(x+size) == max(list_):
                img = TF.crop(img, 0, x+size, h, w-(x+size))
            elif y-size == max(list_):
                img = TF.crop(img, 0, 0, h-(y+size), w)
            elif x-size == max(list_):
                img = TF.crop(img, 0, 0, h, w-(x+size))
            t = transforms.RandomResizedCrop(
                (IMAGE_SIZE), scale=(0.6, 1.0), ratio=(1, 1))
            img = t(img)
        return img


DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
IMAGE_PATH = os.path.join(DATA_PATH, "image")
CSV_PATH = os.path.join(DATA_PATH, "target.csv")
DATASET_PATH = os.path.join(DATA_PATH, "illustFaceDB")
IMAGE_SIZE = 100
split_ratio = 0.9
train_num, val_num = 30000, 3000
rotation_transform = MyRotationTransform([0, 90, 180, 270])
crop_transform = MyRandomCrop()
transform = transforms.Compose([
    rotation_transform,
    transforms.RandomHorizontalFlip(),
])


def fetch_face_data(train_num, val_num, labels):
    train_path = os.path.join(DATASET_PATH, "train", "face")
    val_path = os.path.join(DATASET_PATH, "val", "face")
    os.mkdir(train_path)
    os.mkdir(val_path)
    thd = int(len(labels)*split_ratio)
    selected_list = random.choices(labels[:thd], k=train_num)
    for i, lab in enumerate(tqdm(selected_list)):
        img_path = os.path.join(IMAGE_PATH, "img{}.png".format(lab[0]))
        img = Image.open(img_path)
        img = crop_transform(img, lab, face_flag=True)
        img = transform(img)
        img.save(os.path.join(train_path, "data{}.png".format(i)))
    selected_list = random.choices(labels[thd:], k=val_num)
    for i, lab in enumerate(tqdm(selected_list)):
        img_path = os.path.join(IMAGE_PATH, "img{}.png".format(lab[0]))
        img = Image.open(img_path)
        img = crop_transform(img, lab, face_flag=True)
        img = transform(img)
        img.save(os.path.join(val_path, "data{}.png".format(i)))


def fetch_noface_data(train_num, val_num, labels):
    train_path = os.path.join(DATASET_PATH, "train", "noface")
    val_path = os.path.join(DATASET_PATH, "val", "noface")
    os.mkdir(train_path)
    os.mkdir(val_path)
    thd = int(len(labels)*split_ratio)
    selected_list = random.choices(labels[:thd], k=train_num)
    for i, lab in enumerate(tqdm(selected_list)):
        img_path = os.path.join(IMAGE_PATH, "img{}.png".format(lab[0]))
        img = Image.open(img_path)
        img = crop_transform(img, lab, face_flag=False)
        img = transform(img)
        img.save(os.path.join(train_path, "data{}.png".format(i)))
    selected_list = random.choices(labels[thd:], k=val_num)
    for i, lab in enumerate(tqdm(selected_list)):
        img_path = os.path.join(IMAGE_PATH, "img{}.png".format(lab[0]))
        img = Image.open(img_path)
        img = crop_transform(img, lab, face_flag=False)
        img = transform(img)
        img.save(os.path.join(val_path, "data{}.png".format(i)))


def main():
    csv_file = open(CSV_PATH, 'r', newline="")
    label = list(csv.reader(csv_file, delimiter=","))[1:]
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)
        os.mkdir(os.path.join(DATASET_PATH, "train"))
        os.mkdir(os.path.join(DATASET_PATH, "val"))
        fetch_face_data(train_num, val_num, label)
        fetch_noface_data(train_num, val_num, label)
    else:
        print("illustface_dataset exist!")


if __name__ == "__main__":
    main()
