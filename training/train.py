# %%
import argparse
import matplotlib.pyplot as plt
import torch
import os
import pickle
import yaml
import torchvision.models as models
from utils.predictFaceDataset import predictFaceDataset
from utils.Model import Model
from utils.TrainModel import TrainModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "config.yml"), 'r') as yml:
    config = yaml.load(yml, Loader=yaml.SafeLoader)
    DATA_PATH = os.path.join(BASE_DIR, config['annotation_dataset'])
    DATASET_PATH = os.path.join(BASE_DIR, config['detect_face_dataset'])
IMAGE_PATH = os.path.join(DATA_PATH, "image")
CSV_PATH = os.path.join(DATA_PATH, "face_data.csv")
IMAGE_SIZE = 200
DATA_PATH = os.path.join(os.path.dirname(__file__), "manage_data/data")
DATASET_PATH = os.path.join(DATA_PATH, "predictFaceDB")
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--image_size', type=int, default=100)
parser.add_argument('--weight_name', default='resnet18.pth',
                    help="weight file name")
parser.add_argument('--pretrain_flag', action="store_true")

WEIGHT_DIR = os.path.join(os.path.dirname(__file__), "weight")
pickle_flag = False


def load_dataset(pickle_flag):
    if pickle_flag:
        train_dataset = predictFaceDataset(os.path.join(
            DATASET_PATH, "train"), IMAGE_SIZE)
        test_dataset = predictFaceDataset(os.path.join(
            DATASET_PATH, "val"), IMAGE_SIZE)
        with open(os.path.join(DATA_PATH, "pickle", f"train{IMAGE_SIZE}.pickle"), 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(DATA_PATH, "pickle", f"test{IMAGE_SIZE}.pickle"), 'wb') as f:
            pickle.dump(test_dataset, f)
    else:
        with open(os.path.join(DATA_PATH, "pickle", f"train{IMAGE_SIZE}.pickle"), 'rb') as f:
            train_dataset = pickle.load(f)
        with open(os.path.join(DATA_PATH, "pickle", f"test{IMAGE_SIZE}.pickle"), 'rb') as f:
            test_dataset = pickle.load(f)

    return train_dataset, test_dataset


if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    IMAGE_SIZE = args.image_size
    train_dataset, test_dataset = load_dataset(pickle_flag)
    print(f"train num : {len(train_dataset)}")
    print(f"test num : {len(test_dataset)}")
    print(f"image_size : {IMAGE_SIZE}")
    print(f"batch_size : {batch_size}")
    print(f"model : {args.weight_name}")
    if not args.pretrain_flag:
        print("pretrain : False")
        if args.weight_name == "resnet18":
            model = Model(models.resnet18(pretrained=False))
        elif args.weight_name == "resnet50":
            model = Model(models.resnet50(pretrained=False))
    else:
        print("pretrain : True")
        try:
            if "resnet18" in args.weight_name:
                model = Model(models.resnet18(pretrained=False))
                weight = torch.load(os.path.join(
                    WEIGHT_DIR, args.weight_name))
            elif "resnet50" in args.weight_name:
                model = Model(models.resnet50(pretrained=False))
                weight = torch.load(os.path.join(
                    WEIGHT_DIR, args.weight_name))
        except BaseException:
            print("Not Found weight file")
            exit()
        model.resnet.load_state_dict(weight)

    trainer = TrainModel(model, train_dataset, test_dataset)
    trainer.setting(batch_size, lr)
    for i in range(epochs):
        trainer.train_loop()
        trainer.test_loop()

    train_loss = trainer.train_loss
    test_loss = trainer.test_loss
    trainer.predict_face(10)
    torch.save(trainer.model.state_dict(),
               os.path.join(WEIGHT_DIR, f"{args.model}-{test_loss[-1]}.pth"))
    plt.plot(test_loss)
    plt.savefig(os.path.join(
        DATA_PATH, f"{args.model}-{args.pretrain_flag}-{IMAGE_SIZE}"))
    plt.show()
