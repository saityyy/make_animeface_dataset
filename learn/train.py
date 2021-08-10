# %%
import matplotlib.pyplot as plt
from tqdm import tqdm

from ImageDataset import ImageDataset
from Model import Model
from TrainModel import TrainModel


CSV_PATH = "../../data/target.csv"
IMAGE_PATH = "../../data/image"

train_dataset = ImageDataset(CSV_PATH, IMAGE_PATH, True)
test_dataset = ImageDataset(CSV_PATH, IMAGE_PATH, False)

batch_size = 32
lr = 1e-3
epochs = 30
model = Model()
trainer = TrainModel(model, train_dataset, test_dataset)
trainer.setting(batch_size, lr)
for i in tqdm(range(epochs)):
    trainer.train_loop()
    trainer.test_loop()
loss = trainer.loss
trainer.predict_face()
plt.plot(loss)
plt.show()
