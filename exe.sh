epochs=300
batch_size=256

python train.py --model=resnet50  batch_size=${batch_size} epochs=${epochs}
python train.py --model=resnet18 --pretrained batch_size=${batch_size} epochs=${epochs}
python train.py --model=resnet50 --pretrained batch_size=${batch_size} epochs=${epochs}

