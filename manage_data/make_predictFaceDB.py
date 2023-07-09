import csv
import os
import shutil
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


IMAGE_PATH = os.path.join(os.path.dirname(__file__), "data/image")
CSV_PATH = os.path.join(os.path.dirname(__file__), "data/target.csv")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data/predictFaceDB")
split_ratio = 0.9
ALLOW_ASPECT_RATIO = 1.6


def transform_image(img_path):
    img = cv2.imread(img_path)
    h, w, _ = tuple(img.shape)
    if h > w:
        add_tensor = np.zeros((h, h - w, 3))
        img = np.concatenate([img, add_tensor], axis=1)
    elif w > h:
        add_tensor = np.zeros((w - h, w, 3))
        img = np.concatenate([img, add_tensor], axis=0)
    img = img.astype(np.uint8)
    return img


def aspect_ratio_check(aspect_ratio, img_path) -> bool:
    img = Image.open(img_path)
    wh = [img.width, img.height]
    if max(wh) / min(wh) <= aspect_ratio:
        return True
    else:
        return False


def make_predictFaceDB(face_data):
    data_index_list = []
    for index in range(len(face_data)):
        img_name = f"img{index+1}.png"
        if aspect_ratio_check(ALLOW_ASPECT_RATIO,
                              os.path.join(IMAGE_PATH, img_name)):
            data_index_list.append(index)
    print(len(data_index_list))

    train_img_dir = os.path.join(DATASET_PATH, "train", "image")
    train_face_data_path = os.path.join(DATASET_PATH, "train", "face_data.csv")
    os.mkdir(train_img_dir)
    N = len(data_index_list)
    thd = int(N * split_ratio)
    csv_list = []
    for renban, i in enumerate(tqdm(data_index_list[:thd]), start=1):
        img_name = f"img{i+1}.png"
        src = os.path.join(IMAGE_PATH, img_name)
        dst = os.path.join(train_img_dir, f"img{renban}.png")
        img = transform_image(src)
        cv2.imwrite(dst, img)
        csv_list.append([renban] + face_data[i][1:4])
    with open(train_face_data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "size"])
        writer.writerows(csv_list)

    val_img_dir = os.path.join(DATASET_PATH, "val", "image")
    val_face_data_path = os.path.join(DATASET_PATH, "val", "face_data.csv")
    os.mkdir(val_img_dir)
    csv_list = []
    for renban, i in enumerate(tqdm(data_index_list[thd:]), start=1):
        img_name = f"img{i+1}.png"
        src = os.path.join(IMAGE_PATH, img_name)
        img = transform_image(src)
        dst = os.path.join(val_img_dir, f"img{renban}.png")
        cv2.imwrite(dst, img)
        csv_list.append([renban] + face_data[i][1:4])
    with open(val_face_data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "size"])
        writer.writerows(csv_list)


def main():
    csv_file = open(CSV_PATH, 'r', newline="")
    face_data = list(csv.reader(csv_file, delimiter=","))[1:]
    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)
        os.mkdir(os.path.join(DATASET_PATH, "train"))
        os.mkdir(os.path.join(DATASET_PATH, "val"))
        make_predictFaceDB(face_data)

    else:
        print("predictFaceDB exist!")
        shutil.rmtree(DATASET_PATH)
        main()


if __name__ == "__main__":
    main()
