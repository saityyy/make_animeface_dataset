import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class detectFaceDataset(Dataset):
    def __init__(self, dataset_path, image_size):
        csv_file_path = os.path.join(dataset_path, "face_data.csv")
        self.img_dir = os.path.join(dataset_path, "image")
        self.csv_data = pd.read_csv(csv_file_path).iloc[:, 1:]
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.image = []
        max_wh = []
        for i in tqdm(range(len(os.listdir(self.img_dir)))):
            img_path = os.path.join(
                self.img_dir, f"img{i+1}.png")
            image_ = Image.open(img_path)
            max_wh.append(max(image_.width, image_.height))
            image_ = self.transform(image_)
            self.image.append(image_)
        self.image = torch.stack(self.image)
        self.target_transform(max_wh)

    def target_transform(self, max_wh):
        df = pd.DataFrame({'x1': [], 'y1': [], 'x2': [], 'y2': []}, index=[])
        for i in range(len(os.listdir(self.img_dir))):
            cx = self.csv_data.iloc[i, 0]
            cy = self.csv_data.iloc[i, 1]
            sz = self.csv_data.iloc[i, 2]
            sr = pd.Series([cx - sz, cy - sz, cx + sz, cy + sz],
                           index=["x1", "y1", "x2", "y2"])
            df = pd.concat(
                [df, pd.DataFrame([sr])], ignore_index=True)
            df.iloc[i, :] /= max_wh[i]
        self.face_data = df

    def __len__(self):
        return len(self.face_data)

    def __getitem__(self, idx):
        # img_path = os.path.join(
        # self.img_dir, f"img{idx+1}.png")
        # image = Image.open(img_path)
        # O(1)で画像データを取得
        image = self.image[idx]
        label = torch.tensor(self.face_data.iloc[idx, :].values)
        # if self.transform:
        # image = self.transform(image)
        return image, label

    def image_show(self, idx, predict=None):
        if predict is None:
            _, label = self.__getitem__(idx)
        else:
            label = predict
        img_path = os.path.join(
            self.img_dir, f"img{idx+1}.png")
        image = Image.open(img_path)
        image = image.resize((self.image_size, self.image_size))
        draw = ImageDraw.Draw(image)
        x, y, size = tuple(map(lambda x: x * image.width, list(label)))
        print(x, y, size)
        draw.rectangle((x - size, y - size, x + size, y + size),
                       outline="#000", width=3)
        plt.imshow(image)
        plt.show()
