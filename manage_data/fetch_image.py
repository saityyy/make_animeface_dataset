# python fetch_image.py
# tempフォルダからimageフォルダに移動する。
# imageの番号に合わせてファイル名を変更する。

import os
import shutil

FROMPATH = "../data/temp"
TOPATH = "../data/image"


def fetch_image():
    image_num = len(os.listdir(TOPATH))
    temp_num = len(os.listdir(FROMPATH))
    print("move images")
    print(f"image num : {temp_num}")
    print(f"{image_num} => {image_num+temp_num}")
    for i, img in enumerate(os.listdir(FROMPATH), start=1):
        # 移動先のフォルダに画像をコピー
        src = os.path.join(FROMPATH, img)
        copy = os.path.join(TOPATH, "img{}.png".format(image_num+i))
        shutil.copyfile(src, copy)
        # 移動元の画像を削除
        os.remove(os.path.join(FROMPATH, img))


if __name__ == "__main__":
    fetch_image()
