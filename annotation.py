# python annotation.py
# 一回目のクリックで矩形出現
# 二回目にクリックで矩形の座標やサイズが確定。csvに記録
# fetch_numの数だけimageフォルダから読み込む

import argparse
import os
import csv
import tkinter as tk
from PIL import Image, ImageTk

IMAGEPATH = "../data/image".replace("/", os.sep)
CSVPATH = "../data/target.csv".replace("/", os.sep)
scale = 4  # tkinterで表示する画像の縮小倍率

parser = argparse.ArgumentParser()
parser.add_argument('fetch_num', type=int,
                    help='the number of images you want to fetch')
args = parser.parse_args()
fetch_num = max(args.fetch_num, 1)
root = tk.Tk()

id = None
start_number = sum(1 for _ in open(CSVPATH))+1
centerx, centery = 0, 0
count = 0
size = 0


def click(event):
    global centerx, centery, count, size, _img
    if not centerx == 0:
        index = start_number+count
        with open(CSVPATH, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([index, scale*centerx, scale*centery, scale*size])
            print(count, scale*centerx, scale*centery, scale*size)
        centerx, centery = 0, 0
        count += 1
        if count == fetch_num:
            exit()
        image = os.path.join(
            IMAGEPATH, "img{}.png".format(index+1))
        print(index)
        img = Image.open(image)
        _img = ImageTk.PhotoImage(img.resize(
            (img.width//scale, img.height//scale)))
        canvas.create_image(0, 0, image=_img, anchor=tk.NW)
    else:
        centerx, centery = event.x, event.y


def motion(event):
    global centerx, centery, size, id, img
    if not centerx == 0:
        x, y = centerx, centery
        prev_size = size
        size = abs(centerx-event.x)
        w = img.width//scale
        h = img.height//scale
        window_check = size < x < w-size
        window_check &= size < y < h-size
        if window_check:
            pass
        else:
            size = prev_size
        if id is not None:
            canvas.delete(id)
        id = canvas.create_rectangle(
            x-size, y-size, x+size, y+size)


img_path = os.path.join(IMAGEPATH, "img{}.png".format(start_number))
img = Image.open(img_path)
_img = ImageTk.PhotoImage(img.resize((img.width//scale, img.height//scale)))
canvas = tk.Canvas(
    bg="black", width=4000, height=4000)
canvas.place(x=0, y=0)
canvas.create_image(0, 0, image=_img, anchor=tk.NW)
root.geometry("{}x{}".format(800, 800))
canvas.bind('<Button-1>', click)  # 左クリック
canvas.bind('<Motion>', motion)  # カーソル移動時
root.mainloop()
