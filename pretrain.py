# %%
import os
import shutil
import glob
import random
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.models as models
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# %%
DB_NAME = "illustFaceDB"
DB_PATH = "C:\\Users\sec20\OneDrive\DB\{}".format("illustratorDB")
DB_PATH = os.path.join(os.path.dirname(__file__),
                       "manage_data", "data", "illustFaceDB")
TRAIN_DATASET_PATH = os.path.join(DB_PATH, "train")
VAL_DATASET_PATH = os.path.join(DB_PATH, "val")
WEIGHT_PATH = os.path.join(os.path.dirname(__file__), "weight")
normalize = transforms.Normalize(mean=[0.2, 0.2, 0.2], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.ToTensor(),
])


class Model(nn.Module):
    def __init__(self, pretrained_model):
        super(Model, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )
        self.pretrained_model = pretrained_model

    def forward(self, x):
        x = self.pretrained_model(x)
        x = self.fc(x)
        return x


epoch = 10
batch_size = 256
model_name = "resnet18"
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
train_dataset = ImageFolder(TRAIN_DATASET_PATH, transform)
test_dataset = ImageFolder(VAL_DATASET_PATH, transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(len(train_dataset))
print(len(test_dataset))
if model_name == "resnet50":
    model = Model(models.resnet50(pretrained=False)).to(device)
elif model_name == "resnet18":
    model = Model(models.resnet18(pretrained=False)).to(device)
#wht = torch.load(os.path.join(WEIGHT_PATH, "resnet18.pth"))
# model.pretrained_model.load_state_dict(wht)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)
acc_list = [0]
print(device)
for _ in range(epoch):
    for x, t in tqdm(train_dataloader):
        x, t = x.to(device), t.to(device)
        y = model(x)
        loss = criterion(y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    correct_sum = 0
    for x, t in test_dataloader:
        x, t = x.to(device), t.to(device)
        y = model(x)
        correct_sum += (torch.argmax(y, dim=1) == t).sum()
        loss = criterion(y, t)
    print(correct_sum)
    print(len(test_dataset))
    acc = (correct_sum.cpu().detach().numpy())/len(test_dataset)
    if acc > max(acc_list):
        torch.save(model.pretrained_model.cpu().state_dict(),
                   os.path.join(WEIGHT_PATH, "{}-{}-{:.3g}.pth".format(DB_NAME, model_name, acc)))
        model.pretrained_model.cuda()
    acc_list.append(acc)
    print(acc)
plt.plot(acc_list)
plt.show()


# %%
