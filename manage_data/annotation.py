# python annotation.py
# 一回目のクリックで矩形出現
# 二回目にクリックで矩形の座標やサイズが確定。csvに記録
# fetch_numの数だけimageフォルダから読み込む

import argparse
import os
import csv
import tkinter as tk
from PIL import Image, ImageTk
from fetch_image import fetch_image

IMAGE_PATH = "../data/image"
CSV_PATH = "../data/target.csv"
scale = 6  # tkinterで表示する画像の縮小倍率

parser = argparse.ArgumentParser()
root = tk.Tk()

show_id = None
start_number = sum(1 for _ in open(CSV_PATH))+1
centerx, centery = 0, 0
count = 0
size = 0


def click(event):
    global centerx, centery, count, size, _img, img
    if not centerx == 0:
        index = start_number+count
        with open(CSV_PATH, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([index, scale*centerx, scale*centery, scale*size])
            print(count, scale*centerx, scale*centery, scale*size)
        centerx, centery = 0, 0
        count += 1
        image = os.path.join(IMAGE_PATH, f"img{index+1}.png")
        print(index)
        img = Image.open(image)
        img = img.resize((img.width//scale, img.height//scale))
        _img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, image=_img, anchor=tk.NW)
    else:
        centerx, centery = event.x, event.y


def motion(event):
    global size, show_id, img
    if not centerx == 0:
        x, y = centerx, centery
        prev_size = size
        size = abs(centerx-event.x)
        w = img.width
        h = img.height
        window_check = size < x < w-size
        window_check &= size < y < h-size
        if window_check:
            pass
        else:
            size = prev_size
        if show_id is not None:
            canvas.delete(show_id)
        show_id = canvas.create_rectangle(
            x-size, y-size, x+size, y+size)


fetch_image()
img_path = os.path.join(IMAGE_PATH, "img{}.png".format(start_number))
img = Image.open(img_path)
img = img.resize((img.width//scale, img.height//scale))
_img = ImageTk.PhotoImage(img)
canvas = tk.Canvas(
    bg="black", width=4000, height=4000)
canvas.place(x=0, y=0)
canvas.create_image(0, 0, image=_img, anchor=tk.NW)
root.geometry("{}x{}".format(800, 800))
canvas.bind('<Button-1>', click)  # 左クリック
canvas.bind('<Motion>', motion)  # カーソル移動時
root.mainloop()
