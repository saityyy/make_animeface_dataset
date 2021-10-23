# python check.py
# csvに記録した矩形データが合っているか確認する。
# クリックで次の画像（番号の一つ大きいファイル）を確認する。
# 引数で番号を指定することによって、その番号から確認することができる。
import os
import csv
import argparse
import tkinter as tk
from PIL import Image, ImageTk

IMAGEPATH = "../data/image"
CSVPATH = "../data/target.csv"

parser = argparse.ArgumentParser()
parser.add_argument('start_number', type=int,
                    default="1")

args = parser.parse_args()
start_number = args.start_number
root = tk.Tk()

scale = 3  # tkinterで表示する画像の縮小倍率
count = 0
show_id = None


def click(event):
    global count, img
    count += 1
    next_img_index = start_number+count
    if next_img_index-1 >= len(csv_list):
        print("exit")
        exit()
    image = os.path.join(
        IMAGEPATH, "img{}.png".format(next_img_index))
    img = Image.open(image)
    img = ImageTk.PhotoImage(img.resize((img.width//scale, img.height//scale)))
    canvas.create_image(0, 0, image=img, anchor=tk.NW)
    draw_rectangle(csv_list[next_img_index-1])


def draw_rectangle(data):
    global show_id
    centerx = int(data[1])//scale
    centery = int(data[2])//scale
    size = int(data[3])//scale
    print(data)
    if show_id is not None:
        canvas.delete(show_id)
    show_id = canvas.create_rectangle(
        centerx-size, centery-size, centerx+size, centery+size)


def delete(event):
    # 画像を削除したいときは↓を消す
    click(event)
    number = start_number+count
    end_data = csv_list[-1]
    csv_list[number-1] = end_data
    csv_list[number-1][0] = number
    csv_list.pop()
    total_img = len(os.listdir(IMAGEPATH))
    remove_img_path = os.path.join(IMAGEPATH, "img{}.png".format(number))
    os.remove(remove_img_path)
    os.rename(os.path.join(IMAGEPATH, "img{}.png".format(total_img)),
              remove_img_path)
    with open(CSVPATH, 'w', newline='')as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)
    print("remove {}".format(number))
    exit()


csv_file = open(CSVPATH, "r", newline="")
csv_list = list(csv.reader(csv_file, delimiter=","))
start_number = min(start_number, len(csv_list))
img_path = os.path.join(IMAGEPATH, f"img{start_number}.png")
img = Image.open(img_path)
img = ImageTk.PhotoImage(img.resize((img.width//scale, img.height//scale)))
canvas = tk.Canvas(bg="black", width=2000, height=2000)
canvas.place(x=0, y=0)
canvas.create_image(0, 0, image=img, anchor=tk.NW)
root.geometry("{}x{}".format(800, 800))
draw_rectangle(csv_list[start_number-1])
canvas.bind('<Button-1>', click)  # 左クリック
canvas.bind('<Button-3>', delete)  # 左クリック
root.mainloop()
