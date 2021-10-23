
# %%
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import cv2
import os
import numpy as np

from utils.ImageDataset import ImageDataset
from utils.Model import Model
from utils.TrainModel import TrainModel
from torchvision.transforms import ToTensor
import pickle


CSV_PATH = "../data/target.csv"
IMAGE_PATH = "../data/image"
IMAGE_SIZE = 100
load_path = "./weight/{}".format(os.listdir("./weight")[1])

model = Model()


def main():
    for i in os.listdir(IMAGE_PATH)[:30]:
        img, _ = predict(os.path.join(IMAGE_PATH, i))
        plt.imshow(img)
        plt.show()


def predict(image_path):
    print(load_path)
    img = cv2.imread(os.path.join(image_path))
    h, w, _ = tuple(img.shape)
    if h > w:
        add_tensor = np.zeros((h, h-w, 3))
        img = np.concatenate([img, add_tensor], axis=1)
    elif w > h:
        add_tensor = np.zeros((w-h, w, 3))
        img = np.concatenate([img, add_tensor], axis=0)
    original_size = img.shape[0]
    img = img.astype(np.uint8)
    resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    load_weights = torch.load(load_path)
    model.load_state_dict(load_weights)
    x = ToTensor()(resized_img).reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
    pred = model(x).detach().numpy()[0]
    label = [-1]+list((original_size*pred).astype("int32"))
    pred = tuple((pred*original_size).astype("int32"))
    x, y, size = pred
    print(pred)
    for i in range(2*size):
        try:
            img[y-size, x-size+i] = 0
        except:
            pass
        try:
            img[y+size, x-size+i] = 0
        except:
            pass
        try:
            img[y-size+i, x-size] = 0
        except:
            pass
        try:
            img[y-size+i, x+size] = 0
        except:
            pass
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, label


if __name__ == "__main__":
    main()
