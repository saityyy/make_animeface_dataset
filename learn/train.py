# %%
from ImageDataset import ImageDataset

CSV_PATH = "../../data/target.csv"
IMAGE_PATH = "../../data/image"


train_dataset = ImageDataset(CSV_PATH, IMAGE_PATH, True)
#test_dataset = ImageDataset(CSV_PATH, IMAGE_PATH, False)


# %%
for i in range(10, 30):
    train_dataset.imshow(i)

# %%
