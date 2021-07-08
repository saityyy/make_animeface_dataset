import argparse
import os
import tkinter as tk
from clipImage import ClipImage

FROMPATH = "../Image".replace("/", os.sep)
TOPATH = "./data"
INPUTSIZE = 800
OUTPUTSIZE = 200

parser = argparse.ArgumentParser()
parser.add_argument('fetch_num', type=int,
                    help='the number of images you want to fetch')
args = parser.parse_args()
fetch_num = max(args.fetch_num, 1)
root = tk.Tk()


def main():


if __name__ == '__main__':
    main()
