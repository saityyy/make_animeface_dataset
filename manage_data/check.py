# python check.py
# csvに記録した矩形データが合っているか確認する。
# クリックで次の画像（番号の一つ大きいファイル）を確認する。
# 引数で番号を指定することによって、その番号から確認することができる。
import os
import csv
import argparse
import tkinter as tk
from PIL import Image, ImageTk

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "data/image")
CSVPATH = os.path.join(os.path.dirname(__file__), "data/target.csv")
PAGE_SCALE = 500

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--check_index', type=int,
                    default=f"{len(os.listdir(IMAGE_PATH))-10}", help="select index to start showing.default -> [-10:]")
parser.add_argument('-d', '--delete', type=int, default=-1,
                    help="select index of image to be deleted")
parser.add_argument('-c', '--change_path', default="data",
                    help="you can check or delete predictFaceDB. select 'train' or 'val'")


class Check:
    def __init__(self, start_number, csv_list):
        self.root = tk.Tk()
        self.image_index = min(start_number, len(csv_list))
        self.scale = 1
        self.show_id = None
        self.csv_list = csv_list
        self.canvas = tk.Canvas(bg="black", width=2000, height=2000)

    def __call__(self):
        self.canvas.place(x=0, y=0)
        self.img = self.select_image()
        print(f'image index : {self.image_index}')
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        self.root.geometry("{}x{}".format(PAGE_SCALE, PAGE_SCALE))
        self.draw_rectangle(self.csv_list[self.image_index-1])
        self.canvas.bind('<Button-1>', self.click)  # 左クリック
        self.canvas.bind('<Button-3>', self.click)  # 右クリック
        self.root.mainloop()

    def select_image(self):
        self.img_path = os.path.join(IMAGE_PATH, f"img{self.image_index}.png")
        try:
            img = Image.open(self.img_path)
        except FileNotFoundError:
            self.image_index = 1
            self.img_path = os.path.join(
                IMAGE_PATH, f"img{self.image_index}.png")
            img = Image.open(self.img_path)
        self.scale = max(img.width, img.height)/PAGE_SCALE
        self.img_width = int(img.width/self.scale)
        self.img_height = int(img.height/self.scale)
        img = ImageTk.PhotoImage(
            img.resize((self.img_width, self.img_height)))
        return img

    def click(self, event):
        self.img = self.select_image()
        print(f'image index : {self.image_index}')
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        self.draw_rectangle(self.csv_list[self.image_index-1])

    def draw_rectangle(self, data):
        self.image_index += 1
        data = list(map(int, data))
        cx = int(data[1]/self.scale)
        cy = int(data[2]/self.scale)
        size = int(data[3]/self.scale)
        if self.show_id is not None:
            self.canvas.delete(self.show_id)
        self.show_id = self.canvas.create_rectangle(
            cx-size, cy-size, cx+size, cy+size)


def delete(number, csv_list):
    end_data = csv_list[-1]
    csv_list[number-1] = end_data
    csv_list[number-1][0] = number
    csv_list.pop()
    total_img = len(os.listdir(IMAGE_PATH))
    remove_img_path = os.path.join(IMAGE_PATH, "img{}.png".format(number))
    os.remove(remove_img_path)
    os.rename(os.path.join(IMAGE_PATH, "img{}.png".format(total_img)),
              remove_img_path)
    with open(CSVPATH, 'w', newline='')as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "size"])
        writer.writerows(csv_list)
    print("remove {}".format(number))
    exit()


def get_data_path(sub_dir):
    image_path = os.path.join(os.path.dirname(__file__), sub_dir['image'])
    csv_path = os.path.join(os.path.dirname(__file__), sub_dir['csv'])
    return image_path, csv_path


if __name__ == "__main__":
    args = parser.parse_args()
    if args.change_path == "train" or args.change_path == "val":
        sub_dir = {
            'image': f'data/predictFaceDB/{args.change_path}/image',
            'csv': f'data/predictFaceDB/{args.change_path}/face_data.csv'
        }
        IMAGE_PATH, CSVPATH = get_data_path(sub_dir)
        N = len(os.listdir(IMAGE_PATH))
        args.check_index = args.check_index if args.check_index <= N else N-10
    csv_file = open(CSVPATH, "r", newline="")
    csv_list = list(csv.reader(csv_file, delimiter=","))[1:]
    if args.delete != -1:
        delete(args.delete, csv_list)
    check = Check(args.check_index, csv_list)
    check()
