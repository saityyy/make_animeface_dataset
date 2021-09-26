# python check_face.py
# 指定のフォルダからtempフォルダに画像をコピーする
# データセットとして使えないものを省くために使用する。
# マウス右クリックでtempフォルダに追加、マウス左クリックでスルー。以降、次の画像を表示

import os
import shutil
import random
import string
import csv
import tkinter as tk
from PIL import Image, ImageTk
from predict_method import predict

# 指定のフォルダから画像をランダムに参照する
FROMPATH = "D:/Project/MLProject/DataScience/data/GirlsImage"
TOPATH = "../data/image"
CSV_PATH = "../data/target.csv"

root = tk.Tk()
flist = os.listdir(FROMPATH)
start_number = sum(1 for _ in open(CSV_PATH))+1
print(start_number)
count = 0


def random_name():
    # 重複しないようなランダムなファイル名を生成する
    randlst = [random.choice(string.ascii_letters + string.digits)
               for i in range(10)]
    return ''.join(randlst)


def next_img():
    global img_path, img, label
    img_name = random.choice(flist)
    img_path = os.path.join(FROMPATH, img_name)
    img, label = predict(img_path)
    img = ImageTk.PhotoImage(Image.fromarray(img))
    canvas.create_image(0, 0, image=img, anchor=tk.NW)


def add_image(event):
    global count
    print("count : {}".format(count))
    index = start_number+count
    label[0] = index
    with open(CSV_PATH, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(label)
        print(label)
    src = img_path
    copy = os.path.join(TOPATH, "img{}.png".format(index))
    shutil.copyfile(src, copy)
    count += 1
    print("add")
    next_img()


def no_add(event):
    print("pass")
    next_img()


img_name = random.choice(flist)
img_path = os.path.join(FROMPATH, img_name)
img, label = predict(img_path)
img = ImageTk.PhotoImage(Image.fromarray(img))
canvas = tk.Canvas(
    bg="black", width=2000, height=2000)
canvas.place(x=0, y=0)
canvas.create_image(0, 0, image=img, anchor=tk.NW)
root.geometry("{}x{}".format(400, 400))
canvas.bind('<Button-1>', add_image)  # 左クリック
canvas.bind('<Button-3>', no_add)  # 右クリック
root.mainloop()
