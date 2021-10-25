# 指定のフォルダからtempフォルダに画像をコピーする
# データセットとして使えないものを省くために使用する。
# マウス右クリックでtempフォルダに追加、マウス左クリックでスルー。以降、次の画像を表示

import os
import shutil
import random
import string
import tkinter as tk
from PIL import Image, ImageTk
from make_animefaceDB import aspect_ratio_check

# 指定のフォルダから画像をランダムに参照する
FROMPATH = "D:/Project/MLProject/DataScience/data/GirlsImage"
TOPATH = os.path.join(os.path.dirname(__file__), "data/temp")
allow_aspect_ratio = 1.4


class CheckFace:
    def __init__(self):
        self.root = tk.Tk()
        self.image_scandir_list = list(os.scandir(FROMPATH))
        self.canvas = tk.Canvas(
            bg="black", width=2000, height=2000)
        self.add_image_count = 0
        self.image_num = len(os.listdir(TOPATH))

    @property
    def current_image_num(self):
        return self.add_image_count+self.image_num

    def __call__(self):
        self.canvas.place(x=0, y=0)
        self.img = self.select_image()
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        self.root.geometry("{}x{}".format(800, 800))
        self.canvas.bind('<Button-1>', self.add_image)  # 左クリック
        self.canvas.bind('<Button-3>', self.no_add)  # 右クリック
        self.root.mainloop()

    def select_image(self):
        while(True):
            self.img_path = random.choice(self.image_scandir_list).path
            if aspect_ratio_check(allow_aspect_ratio, self.img_path):
                break
        img = Image.open(self.img_path)
        img = ImageTk.PhotoImage(
            img.resize((img.width//5, img.height//5)))
        return img

    # 重複しないようなランダムなファイル名を生成する
    def random_name(self):
        randlst = [random.choice(string.ascii_letters + string.digits)
                   for i in range(10)]
        return ''.join(randlst)

    def next_img(self):
        self.img = self.select_image()
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)

    def add_image(self, event):
        src = self.img_path
        copy = os.path.join(TOPATH, self.random_name()+".png")
        shutil.copyfile(src, copy)
        print("add")
        self.add_image_count += 1
        self.next_img()

    def no_add(self, event):
        print("pass")
        self.next_img()


if __name__ == "__main__":
    cf = CheckFace()
    try:
        cf()
    finally:
        print("add image : {}".format(cf.add_image_count))
        print("image num : {} -> {}".format(cf.image_num, cf.current_image_num))
