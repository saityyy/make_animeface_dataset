# %%
import matplotlib.pyplot as plt
import torch
import os
import pickle
from utils.ImageDataset import ImageDataset
from utils.Model import Model
from utils.TrainModel import TrainModel


IMAGE_SIZE = 200
DATASET_PATH = os.path.join(os.path.dirname(
    __file__), "manage_data/data/predictFaceDB")
load_path = "./weight/resnet18.pth"
load_path = "./weight/{}".format(os.listdir("./weight")[1])
print(load_path)

pickle_flag = False
if pickle_flag:
    train_dataset = ImageDataset(os.path.join(
        DATASET_PATH, "train"), IMAGE_SIZE)
    test_dataset = ImageDataset(os.path.join(
        DATASET_PATH, "val"), IMAGE_SIZE)
    with open('train_dataset.pickle', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('test_dataset.pickle', 'wb') as f:
        pickle.dump(test_dataset, f)
else:
    with open('train_dataset.pickle', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('test_dataset.pickle', 'rb') as f:
        test_dataset = pickle.load(f)

print(f"train num : {len(train_dataset)}")
print(f"test num : {len(test_dataset)}")
batch_size = 128
lr = 1e-4
epochs = 1

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
#torch.save(trainer.model.state_dict(), f"./weight/weight{test_loss[-1]}.pt")
plt.plot(test_loss)
plt.show()
