# python fetch_image.py
# tempフォルダからimageフォルダに移動する。
# imageの番号に合わせてファイル名を変更する。

import os
import shutil

FROMPATH = "../data/temp".replace("/", os.sep)
TOPATH = "../data/image".replace("/", os.sep)

image_num = len(os.listdir(TOPATH))
temp_num = len(os.listdir(FROMPATH))
print("move images")
print("image num : {}".format(temp_num))
print("{} => {}".format(image_num, image_num+temp_num))
for i, img in enumerate(os.listdir(FROMPATH)):
    # 移動先のフォルダに画像をコピー
    src = os.path.join(FROMPATH, img)
    copy = os.path.join(TOPATH, "img{}.png".format(image_num+i+1))
    shutil.copyfile(src, copy)
    # 移動元の画像を削除
    os.remove(os.path.join(FROMPATH, img))
