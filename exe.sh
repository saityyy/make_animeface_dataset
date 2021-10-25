epochs=300
batch_size=256

python train.py --model=resnet50  batch_size=${batch_size} epochs=${epochs}
python train.py --model=resnet18 --pretrain_flag batch_size=${batch_size} epochs=${epochs}
python train.py --model=resnet50 --pretrain_flag batch_size=${batch_size} epochs=${epochs}

