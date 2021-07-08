import os

IMAGEPATH = "../data/image".replace("/", os.sep)

for i, img in enumerate(os.listdir(IMAGEPATH)):
    from_path = os.path.join(IMAGEPATH, img)
    to_path = os.path.join(IMAGEPATH, "img{}.png".format(i+1))
    os.rename(from_path, to_path)
