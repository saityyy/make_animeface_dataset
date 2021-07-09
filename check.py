# python check.py
# csvに記録した矩形データが合っているか確認する。
# クリックで次の画像（番号の一つ大きいファイル）を確認する。
# 引数で番号を指定することによって、その番号から確認することができる。
import os
import csv
import argparse
import tkinter as tk
from PIL import Image, ImageTk

IMAGEPATH = "../data/image".replace("/", os.sep)
CSVPATH = "../data/target.csv".replace("/", os.sep)

parser = argparse.ArgumentParser()
parser.add_argument('start_number', type=int,
                    default="1")

args = parser.parse_args()
start_number = args.start_number
root = tk.Tk()

scale = 3  # tkinterで表示する画像の縮小倍率
count = 0
id = None


def click(event):
    global count, img, f
    count += 1
    next_img = start_number+count
    if next_img-1 >= len(f):
        print("exit")
        exit()
    image = os.path.join(
        IMAGEPATH, "img{}.png".format(next_img))
    img = Image.open(image)
    img = ImageTk.PhotoImage(img.resize((img.width//scale, img.height//scale)))
    canvas.create_image(0, 0, image=img, anchor=tk.NW)
    draw_rectangle(f[next_img-1])


def draw_rectangle(data):
    global id
    centerx = int(data[1])//scale
    centery = int(data[2])//scale
    size = int(data[3])//scale
    print(data)
    if id is not None:
        canvas.delete(id)
    id = canvas.create_rectangle(
        centerx-size, centery-size, centerx+size, centery+size)


csv_file = open(CSVPATH, "r", newline="")
f = list(csv.reader(csv_file, delimiter=","))
img_path = os.path.join(IMAGEPATH, "img{}.png".format(start_number))
img = Image.open(img_path)
img = ImageTk.PhotoImage(img.resize((img.width//scale, img.height//scale)))
canvas = tk.Canvas(
    bg="black", width=2000, height=2000)
canvas.place(x=0, y=0)
canvas.create_image(0, 0, image=img, anchor=tk.NW)
root.geometry("{}x{}".format(800, 800))
draw_rectangle(f[start_number-1])
canvas.bind('<Button-1>', click)  # 左クリック
root.mainloop()
