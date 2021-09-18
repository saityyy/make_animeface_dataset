# %%
import pickle

with open("./test_dataset.pickle", 'rb')as f:
    test_dataset = pickle.load(f)

print(test_dataset.img.shape)
# %%
for i in range(4):
    test_dataset.imshow(i)
# %%
