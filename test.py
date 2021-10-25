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


class test:
    def __init__(self):
        self.a = list(range(10))

    def __getitem__(self, idx):
        return os.path.join("./test", f"{self.a[idx]}.txt")


a = test()
print(a[:3])
