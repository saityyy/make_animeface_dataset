from torch.utils import data
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

IMAGE_SIZE = 200
train_data_ratio = 0.9


class ImageDataset(Dataset):
    def __init__(self, csv_file_path, img_dir, train_flag, transform=ToTensor()):
        self.csv_data = np.loadtxt(csv_file_path, delimiter=',')
        self.csv_data = np.delete(self.csv_data, 0, 1)
        self.img = []
        self.img_labels = []
        data_num = len(self.csv_data)
        partition = max(int(data_num*train_data_ratio), data_num-300)
        print(partition)
        if train_flag:
            iter = range(0, partition, 1)
        else:
            iter = range(partition, data_num, 1)
        for i in tqdm(iter):
            fetch_img = cv2.imread(os.path.join(img_dir, f"img{i+1}.png"))
            resize_img, label = self.clip_images(
                fetch_img, self.csv_data[i])
            self.img.append(resize_img)
            self.img_labels.append(label)
        self.img = np.array(self.img, dtype="float32")
        self.img_labels = np.array(self.img_labels, dtype="float32")
        self.img_labels /= IMAGE_SIZE
        self.transform = transform
        print(self.img.shape)
        print(self.img_labels.shape)

    def clip_images(self, img, label):
        h, w, _ = tuple(img.shape)
        if h > w:
            add_tensor = np.zeros((h, h-w, 3))
            img = np.concatenate([img, add_tensor], axis=1)
        elif w > h:
            add_tensor = np.zeros((w-h, w, 3))
            img = np.concatenate([img, add_tensor], axis=0)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        new_label = list(map(lambda x: int(x*IMAGE_SIZE/max(h, w)), label))
        return (img, new_label)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def imshow(self, idx, pred=None):
        image = cv2.cvtColor(self.img[idx], cv2.COLOR_BGR2RGB)
        image = image.astype("int32")
        x, y, size = tuple((self.img_labels[idx]*IMAGE_SIZE).astype("int32"))
        if pred is not None:
            print(pred)
            pred = tuple((pred*IMAGE_SIZE).astype("int32"))
            x, y, size = pred
        print(x, y, size)
        for i in range(2*size):
            try:
                image[y-size, x-size+i] = 0
            except:
                pass
            try:
                image[y+size, x-size+i] = 0
            except:
                pass
            try:
                image[y-size+i, x-size] = 0
            except:
                pass
            try:
                image[y-size+i, x+size] = 0
            except:
                pass
        plt.imshow(image)
        plt.show()
