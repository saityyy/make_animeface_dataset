# tempフォルダーから画像を取ってきてアノテーションをする。
# 一回目のクリックで矩形出現
# 二回目にクリックで矩形の座標やサイズが確定。csvに記録

import shutil
import os
import csv
import tkinter as tk
from PIL import Image, ImageTk

from manage_data.check import IMAGEPATH

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
TEMP_PATH = os.path.join(DATA_PATH, "temp")
IMAGE_PATH = os.path.join(DATA_PATH, "image")
CSV_PATH = os.path.join(DATA_PATH, "target.csv")


class Annotation:
    def __init__(self):
        if not os.path.isdir(DATA_PATH):
            os.mkdir(DATA_PATH)
            os.mkdir(TEMP_PATH)
            os.mkdir(IMAGEPATH)
            with open(CSV_PATH, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "x", "y", "size"])
        self.show_id = None
        self.start_index = sum(1 for _ in open(CSV_PATH))-1
        print(self.start_index)
        self.img_index = self.start_index+1
        self.centerx, self.centery, self.size = 0, 0, 0
        self.first_temp_num = len(os.listdir(TEMP_PATH))
        self.scale = 1
        self.root = tk.Tk()
        self.image_scandir_iter = os.scandir(TEMP_PATH)
        self.canvas = tk.Canvas(
            bg="black", width=4000, height=4000)

    @property
    def current_temp_num(self):
        return len(os.listdir(TEMP_PATH))

    @property
    def current_image_num(self):
        return len(os.listdir(IMAGE_PATH))

    def __call__(self):
        self.canvas.place(x=0, y=0)
        self.img = self.select_image()
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        self.root.geometry("{}x{}".format(800, 800))
        self.canvas.bind('<Button-1>', self.click)  # 左クリック
        self.canvas.bind('<Motion>', self.motion)  # カーソル移動時
        self.root.mainloop()

    # アノテーションした画像をIMAGEPATHに移動させる。
    def temp2image(self):
        src = self.img_path
        dst = os.path.join(IMAGE_PATH, "img{}.png".format(self.img_index))
        shutil.copyfile(src, dst)
        os.remove(self.img_path)

    # tempフォルダーの画像を一枚読み込む。
    def select_image(self):
        try:
            self.img_path = next(self.image_scandir_iter).path
        except StopIteration:
            print("temp folder is empty")
            exit()
        img = Image.open(self.img_path)
        self.scale = max(img.width, img.height)/500
        self.img_width = int(img.width/self.scale)
        self.img_height = int(img.height/self.scale)
        print(img.width, img.height)
        print(self.img_width, self.img_height)
        img = ImageTk.PhotoImage(
            img.resize((self.img_width, self.img_height)))
        return img

    def click(self, event):
        if not self.centerx == 0:
            cx, cy, size = tuple(
                map(int, [self.scale*self.centerx, self.scale*self.centery, self.scale*self.size]))
            with open(CSV_PATH, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [self.img_index, cx, cy, size])
                print(self.img_index, cx, cy, size)
            self.centerx, self.centery, self.size = 0, 0, 0
            self.temp2image()
            self.img_index += 1
            self.img = self.select_image()
            self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        else:
            self.centerx, self.centery = event.x, event.y

    def motion(self, event):
        if not self.centerx == 0:
            x, y, size = self.centerx, self.centery, self.size
            new_size = abs(self.centerx-event.x)
            w, h = self.img_width, self.img_height
            window_check = new_size < x < w-new_size
            window_check &= new_size < y < h-new_size
            if window_check:
                self.size = new_size
            else:
                self.size = size
            if self.show_id is not None:
                self.canvas.delete(self.show_id)
            self.show_id = self.canvas.create_rectangle(
                x-size, y-size, x+size, y+size)


if __name__ == "__main__":
    at = Annotation()
    try:
        at()
    finally:
        print(f"image num : {at.start_index} -> {at.current_image_num}")
        print(f"temp num : {at.first_temp_num} -> {at.current_temp_num}")
