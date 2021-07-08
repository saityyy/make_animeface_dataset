import os
import shutil
import random
import string
import tkinter as tk
from PIL import Image, ImageTk

FROMPATH = ("D:/Project/MLProject/DataScience/data/GirlsImage"
            .replace("/", os.sep))
TOPATH = "../data/temp".replace("/", os.sep)

root = tk.Tk()
flist = os.listdir(FROMPATH)


def random_name():
    randlst = [random.choice(string.ascii_letters + string.digits)
               for i in range(10)]
    return ''.join(randlst)


def next_img():
    global img, img_path, flist
    img_name = random.choice(flist)
    img_path = os.path.join(FROMPATH, img_name)
    img = Image.open(img_path)
    img = ImageTk.PhotoImage(img.resize((img.width//5, img.height//5)))
    canvas.create_image(0, 0, image=img, anchor=tk.NW)


def add_image(event):
    global img_path
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
