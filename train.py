# %%
import matplotlib.pyplot as plt
import torch
import os
import pickle

from utils.ImageDataset import ImageDataset
from utils.Model import Model
from utils.TrainModel import TrainModel


CSV_PATH = "../data/target.csv"
IMAGE_PATH = "../data/image"
load_path = "./weight/resnet18.pth"
load_path = "./weight/{}".format(os.listdir("./weight")[1])
print(load_path)

make_dataset_flag = True
# pickleデータでデータセット読み込み
if make_dataset_flag:
    train_dataset = ImageDataset(CSV_PATH, IMAGE_PATH, True)
    test_dataset = ImageDataset(CSV_PATH, IMAGE_PATH, False)
    with open('./train_dataset.pickle', 'wb')as f:
        pickle.dump(train_dataset, f)
    with open('./test_dataset.pickle', 'wb')as f:
        pickle.dump(test_dataset, f)
else:
    with open('./train_dataset.pickle', 'rb')as f:
        train_dataset = pickle.load(f)
    with open('./test_dataset.pickle', 'rb')as f:
        test_dataset = pickle.load(f)
    print(train_dataset.img.shape)
    print(test_dataset.img.shape)
# データセットを１から作成

batch_size = 128
lr = 1e-4
epochs = 300

model = Model()
weight = torch.load(load_path)
# model.resnet18.load_state_dict(weight)
model.load_state_dict(weight)
trainer = TrainModel(model, train_dataset, test_dataset)
trainer.setting(batch_size, lr)
for i in range(epochs):
    trainer.train_loop()
    trainer.test_loop()

train_loss = trainer.train_loss
test_loss = trainer.test_loss
trainer.predict_face(10)
torch.save(trainer.model.state_dict(), f"./weight/weight{test_loss[-1]}.pt")
plt.plot(test_loss)
plt.show()
