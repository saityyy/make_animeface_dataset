from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from tqdm import tqdm

IMAGE_SIZE = 400
train_data_ratio = 0.8


class ImageDataset(Dataset):
    def __init__(self, csv_file_path, img_dir, train_flag, transform=ToTensor()):
        self.csv_data = np.loadtxt(csv_file_path, delimiter=',')
        self.csv_data = np.delete(self.csv_data, 0, 1)
        print(self.csv_data.shape)
        self.img = []
        self.img_labels = []
        for i in tqdm(range(len(self.csv_data))):
            fetch_img = cv2.imread(os.path.join(img_dir, f"img{i+1}.png"))
            resize_img, label = self.clip_images(fetch_img, self.csv_data[i])
            self.img.append(resize_img)
            self.img_labels.append(label)
        self.img = np.array(self.img, dtype="float32")
        self.img_labels = np.array(self.img_labels, dtype="float32")
        self.img_labels /= self.img.shape[0]
        print(self.img_labels.shape)
        # 訓練データとテストデータに分ける
        partition = int(len(self.img)*train_data_ratio)
        print(self.img.shape)
        print(self.img_labels.shape)
        print(partition)
        if train_flag:
            self.img = self.img[:partition]
            self.img_labels = self.img_labels[:partition]
        else:
            self.img = self.img[partition:]
            self.img_labels = self.img_labels[partition:]
        self.transform = transform
        print(self.img.shape)
        print(self.img_labels.shape)

    def clip_images(self, img, label):
        x, y, size = tuple(label)
        h, w, _ = tuple(img.shape)
        new_label = None
        if h >= w:
            new_img_x = 0
            start = max(y+size-w, 0)
            end = y-size
            new_img_y = random.randrange(start, end)
            a = int(w/2)
            new_label = [x, y-new_img_y, size]
        else:
            new_img_y = 0
            start = max(x+size-h, 0)
            end = x-size
            new_img_x = random.randrange(start, end)
            a = int(h/2)
            new_label = [x-new_img_x, y, size]
        img = img[new_img_y:new_img_y+2*a, new_img_x:new_img_x+2*a]
        new_label = list(map(lambda x: int(x*IMAGE_SIZE/len(img)), new_label))
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        return (img, new_label)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def imshow(self, idx):
        img_ = cv2.cvtColor(self.img[idx], cv2.COLOR_BGR2RGB)
        plt.imshow(img_)
        plt.show()
