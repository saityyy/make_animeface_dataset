# %%
import argparse
import torch
import os
import yaml
from pprint import pprint
import torchvision.models as models
from utils.detectFaceDataset import detectFaceDataset
from utils.Model import Model
from utils.TrainModel import TrainModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, "config.yml"), 'r') as yml:
    config = yaml.load(yml, Loader=yaml.SafeLoader)
    DATASET_PATH = os.path.join(BASE_DIR, config['detect_face_dataset'])
WEIGHT_DIR = os.path.join(BASE_DIR, "weight")
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--image_size', type=int, default=300)
parser.add_argument('--weight_name', default='resnet18',
                    help="weight file name")
parser.add_argument('--pretrain_flag', action="store_true")


def load_dataset():
    train_dataset = detectFaceDataset(os.path.join(
        DATASET_PATH, "train"), IMAGE_SIZE)
    test_dataset = detectFaceDataset(os.path.join(
        DATASET_PATH, "val"), IMAGE_SIZE)
    return train_dataset, test_dataset


if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    IMAGE_SIZE = args.image_size
    train_dataset, test_dataset = load_dataset()
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
    pprint(test_loss)
    torch.save(trainer.model.state_dict(),
               os.path.join(WEIGHT_DIR, f"{args.weight_name}_{batch_size}_{IMAGE_SIZE}_{lr}_{test_loss[-1]}.pth"))
