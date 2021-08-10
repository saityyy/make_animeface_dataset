# python check_face.py
# 指定のフォルダからtempフォルダに画像をコピーする
# データセットとして使えないものを省くために使用する。
# マウス右クリックでtempフォルダに追加、マウス左クリックでスルー。以降、次の画像を表示

import os
import shutil
import random
import string
import tkinter as tk
from PIL import Image, ImageTk

# 指定のフォルダから画像をランダムに参照する
FROMPATH = "D:/Project/MLProject/DataScience/data/GirlsImage"
TOPATH = "../data/temp"

root = tk.Tk()
flist = os.listdir(FROMPATH)


def random_name():
    # 重複しないようなランダムなファイル名を生成する
    randlst = [random.choice(string.ascii_letters + string.digits)
               for i in range(10)]
    return ''.join(randlst)


def next_img():
    global img, img_path
    img_name = random.choice(flist)
    img_path = os.path.join(FROMPATH, img_name)
    img = Image.open(img_path)
    img = ImageTk.PhotoImage(img.resize((img.width//5, img.height//5)))
    canvas.create_image(0, 0, image=img, anchor=tk.NW)


def add_image(event):
    #global img_path
    src = img_path
    copy = os.path.join(TOPATH, random_name()+".png")
    shutil.copyfile(src, copy)
    print("add")
    next_img()


def no_add(event):
    print("pass")
    next_img()


img_name = random.choice(flist)
img_path = os.path.join(FROMPATH, img_name)
img = Image.open(img_path)
img = ImageTk.PhotoImage(img.resize((img.width//5, img.height//5)))
canvas = tk.Canvas(
    bg="black", width=2000, height=2000)
canvas.place(x=0, y=0)
canvas.create_image(0, 0, image=img, anchor=tk.NW)
root.geometry("{}x{}".format(800, 800))
canvas.bind('<Button-1>', add_image)  # 左クリック
canvas.bind('<Button-3>', no_add)  # 右クリック
root.mainloop()
