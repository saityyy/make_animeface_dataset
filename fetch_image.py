import os
import shutil

FROMPATH = "../data/temp".replace("/", os.sep)
TOPATH = "../data/image".replace("/", os.sep)

image_num = len(os.listdir(TOPATH))
print(image_num)
for i, img in enumerate(os.listdir(FROMPATH)):
    src = os.path.join(FROMPATH, img)
    copy = os.path.join(TOPATH, "img{}.png".format(image_num+i+1))
    shutil.copyfile(src, copy)
    os.remove(os.path.join(FROMPATH, img))
