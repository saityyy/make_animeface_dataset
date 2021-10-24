# tempフォルダーから画像を取ってきてアノテーションをする。
# 一回目のクリックで矩形出現
# 二回目にクリックで矩形の座標やサイズが確定。csvに記録

import argparse
import shutil
import os
import csv
import tkinter as tk
from PIL import Image, ImageTk

TEMP_PATH = os.path.join(os.path.dirname(__file__), "data/temp")
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "data/image")
CSV_PATH = os.path.join(os.path.dirname(__file__), "data/target.csv")
scale = 6  # tkinterで表示する画像の縮小倍率

parser = argparse.ArgumentParser()


class Annotation:
    def __init__(self, scale):
        self.show_id = None
        self.start_index = sum(1 for _ in open(CSV_PATH))+1
        self.img_index = self.start_index
        self.centerx, self.centery, self.size = 0, 0, 0
        self.first_temp_num = len(os.listdir(TEMP_PATH))
        self.scale = scale
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
        self.img_width, self.img_height = img.width, img.height
        img = ImageTk.PhotoImage(
            img.resize((img.width//scale, img.height//scale)))
        return img

    def click(self, event):
        if not self.centerx == 0:
            cx, cy, size = self.centerx, self.centery, self.size
            with open(CSV_PATH, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [self.img_index, scale*cx, scale*cy, scale*size])
                print(self.img_index, scale*cx, scale*cy, scale*size)
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
    at = Annotation(scale)
    try:
        at()
    finally:
        print(f"image num : {at.start_index} -> {at.current_image_num}")
        print(f"temp num : {at.first_temp_num} -> {at.current_temp_num}")
