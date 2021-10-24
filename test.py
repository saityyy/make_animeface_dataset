import os

path = "./manage_data"
a = os.scandir(path)
print(type(a))
print(a.__next__().name)
print(a.__next__().name)
print(a.__next__().name)
print(a.__next__().name)
print(a.__next__().name)
print(a.__next__().name)
print(a.__next__().name)
print(list(a))
