import os
import csv
import tkinter as tk
from PIL import Image, ImageTk

IMAGEPATH = "../data/image".replace("/", os.sep)
CSVPATH = "../data/target.csv".replace("/", os.sep)
INPUTSIZE = 800

root = tk.Tk()

start_number = 1
count = 0
id = None


def click(event):
    global count, img, f
    count += 1
    next_img = start_number+count
    image = os.path.join(
        IMAGEPATH, "img{}.png".format(next_img))
    img = Image.open(image)
    img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=img, anchor=tk.NW)
    draw_rectangle(f[next_img-1])


def draw_rectangle(data):
    global id
    centerx = int(data[1])
    centery = int(data[2])
    size = int(data[3])
    print(data)
    if id is not None:
        canvas.delete(id)
    id = canvas.create_rectangle(
        centerx-size, centery-size, centerx+size, centery+size)


csv_file = open(CSVPATH, "r", newline="")
f = list(csv.reader(csv_file, delimiter=","))
img_path = os.path.join(IMAGEPATH, "img{}.png".format(start_number))
img = Image.open(img_path)
img = ImageTk.PhotoImage(img)
canvas = tk.Canvas(
    bg="black", width=2000, height=2000)
canvas.place(x=0, y=0)
canvas.create_image(0, 0, image=img, anchor=tk.NW)
root.geometry("{}x{}".format(800, 800))
draw_rectangle(f[start_number-1])
canvas.bind('<Button-1>', click)  # 左クリック
root.mainloop()
