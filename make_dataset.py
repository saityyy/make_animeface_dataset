import csv
import os
import random
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from tqdm import tqdm


class MakeDataset:
    def __init__(self, root, images_path, moveto_path, fetch_num, img_size):
        self.centerx = 0
        self.centery = 0
        self.size = 0
        self.id = None
        self.root = root
        self.img_size = int(img_size)
        if not os.path.exists(moveto_path):
            os.mkdir(moveto_path)
            os.mkdir(os.path.join(moveto_path, "images"))
        self.root_path = moveto_path
        self.moveto_path = os.path.join(moveto_path, "images")
        self.count = 0
        self.csvfile_path = os.path.join(moveto_path, "target.csv")
        self.falsecsv_path = os.path.join(moveto_path, "false.csv")
        if not os.path.isfile(self.csvfile_path):
            _ = open(self.csvfile_path, 'w')
        self.start_number = sum(1 for _ in open(self.csvfile_path))+1
        for i in range(self.start_number, len(os.listdir(self.moveto_path))+1):
            os.remove(os.path.join(self.moveto_path, "img{}.png".format(i)))

        self.images_path = images_path
        if not os.path.exists(self.images_path):
            print("Incorrect path")
            exit()
        self.fetch_num = fetch_num

    #
    def fetch_data(self):
        images_num = len(os.listdir(self.images_path))
        fetch_num = min(self.fetch_num, images_num)
        print("number of images : {}".format(images_num))
        print("fetch images : {}".format(fetch_num))
        count = 0
        file_list = os.listdir(self.images_path)
        random.shuffle(file_list)
        for i, file in enumerate(file_list):
            image = cv2.imread(os.path.join(self.images_path, file))
            w, h = float(image.shape[1]), float(image.shape[0])
            if file[-3:] == "gif" or h < w:
                continue
            image = image[0:int(w)][0:int(w)]
            save_path = os.path.join(
                self.moveto_path, "img{}.png".format(self.start_number+count))
            cv2.imwrite(save_path, cv2.resize(
                image, (self.img_size, self.img_size)))
            count += 1
            if count == fetch_num:
                break
    # tkinterライブラリを用いて画像を表示,fetch_num回数分アノテーションを行う

    def annotate_image(self):
        self.img_path = os.path.join(
            self.moveto_path, "img{}.png".format(self.start_number))
        self.img = Image.open(self.img_path)
        self.img = ImageTk.PhotoImage(self.img)
        self.canvas = tk.Canvas(
            bg="black", width=self.img_size, height=self.img_size)
        self.canvas.place(x=0, y=0)
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        self.root.geometry("{}x{}".format(self.img_size, self.img_size))
        self.canvas.bind('<Button-1>', self.click)  # 左クリック
        self.canvas.bind('<Button-3>', self.delete_img)  # 右クリック
        self.canvas.bind('<Motion>', self.motion)  # カーソル移動時
        self.canvas.pack()
        self.root.mainloop()

    # クリックしたときの画面遷移,csv書き込みなど
    def click(self, event):
        if not self.centerx == 0:
            count = self.start_number+self.count
            with open(self.csvfile_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([count, self.centerx, self.centery, self.size])
                print(count, self.centerx, self.centery, self.size)
            self.centerx, self.centery = 0, 0
            self.count += 1
            if self.count == self.fetch_num:
                self.make_falsedataset()
                exit()
            self.image = os.path.join(
                self.moveto_path, "img{}.png".format(count+1))
            self.img = Image.open(self.image)
            self.img = ImageTk.PhotoImage(self.img)
            self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        else:
            self.centerx, self.centery = event.x, event.y

    # 顔領域を指定するときのウィンドウを可視化する
    def motion(self, event):
        if not self.centerx == 0:
            x, y = self.centerx, self.centery
            prev_size = self.size
            self.size = abs(self.centerx-event.x)
            window_check = self.size < x < self.img_size-self.size
            window_check &= self.size < y < self.img_size-self.size
            if window_check:
                pass
            else:
                self.size = prev_size
            if self.id is not None:
                self.canvas.delete(self.id)
            self.id = self.canvas.create_rectangle(
                x-self.size, y-self.size, x+self.size, y+self.size)

    # 右クリックでデータセット対象の画像からはずす
    def delete_img(self, event):
        print(self.start_number+self.count, "delete")
        with open(self.csvfile_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.start_number+self.count, 0, 0, 0])
        self.count += 1
        if self.count == self.fetch_num:
            exit()
        self.image_path = os.path.join(
            self.moveto_path, "img{}.png".format(self.count+self.start_number))
        self.img = Image.open(self.image_path)
        self.img = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)

    # 画像をランダムに切り抜くメソッド
    def make_falsedataset(self):
        total_num = len(os.listdir(self.moveto_path))
        with open(self.falsecsv_path, 'w', newline='')as f:
            for i in range(total_num):
                writer = csv.writer(f)
                x = random.randint(50, 750)
                y = random.randint(50, 750)
                max_size = min([x, y, 800-x, 800-y])
                min_size = 50
                size = random.randint(min_size, max_size)
                writer.writerow([i+1, x, y, size])

    # 顔画像とそうでないものを分類するためのデータセットを作成するメソッド
    def make_classify_dataset(self):
        total_num = range(1, int(len(os.listdir(self.moveto_path))+1))
        total_num = range(1, self.start_number)
        face_images_path = os.path.join(self.root_path, "face_images")
        random_trim_images_path = os.path.join(
            self.root_path, "random_trim_images")
        if not os.path.exists(face_images_path):
            os.mkdir(face_images_path)
            os.mkdir(random_trim_images_path)
        # 画像をcsvファイルの数値に従って切り抜く。リサイズされたあとの画像は200x200
        num = 0
        print("download...")
        for i in tqdm(total_num):
            img_path = os.path.join(self.moveto_path, "img{}.png".format(i))
            img = cv2.imread(img_path)
            # face_imageを保存
            fetched_line = np.loadtxt(self.csvfile_path, delimiter=',')[i-1]
            number, x, y, size = tuple(map(int, fetched_line))
            if x == 0:
                continue
            img_trim = img[y-size:y+size, x-size:x+size]
            save_path = os.path.join(
                face_images_path, "face{}.png".format(num))
            cv2.imwrite(save_path, cv2.resize(img_trim, (200, 200)))
            # random_trim_imagesを保存
            fetched_line = np.loadtxt(self.falsecsv_path, delimiter=',')[i-1]
            number, x, y, size = tuple(map(int, fetched_line))
            img_trim = img[y-size:y+size, x-size:x+size]
            save_path = os.path.join(
                random_trim_images_path, "random{}.png".format(num))
            cv2.imwrite(save_path, cv2.resize(img_trim, (200, 200)))
            num += 1
