from torchvision.transforms import ToTensor
import torch
from torchvision.io import read_image
import os
import pandas as pd

path = "./manage_data/data/target.csv"
df = pd.read_csv(path, index_col=0)
print(df.iloc[0, 0])
# print(df.loc['index'])
df.iloc[0, :] /= 10000
print(df.head())
print(df.columns)
print(df.loc[:, 'x'])
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})


a = torch.zeros((5, 6))
b = []
for _ in range(7):
    b.append(a)
print(b)
b = torch.stack(b)
print(b.shape)
