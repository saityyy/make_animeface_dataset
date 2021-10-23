from torch import nn
import torch
import torchvision.models as models


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.loss = []
        self.acc = []
        self.resnet18 = models.resnet18(pretrained=False)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)
        for i in range(len(x)):
            outx = x[i][0].clone()
            outy = x[i][1].clone()
            out_size = x[i][2].clone()
            outx = max(min(outx+0.5, 1), 0)
            outy = max(min(outy+0.5, 1), 0)
            out_size = abs(out_size)
            if(outx+out_size > 1):
                out_size = 1-outx
            if(outy+out_size > 1):
                out_size = 1-outy
            if(outx-out_size < 0):
                out_size = outx
            if(outy-out_size < 0):
                out_size = outy
            x[i][0] = outx
            x[i][1] = outy
            x[i][2] = out_size

        return x


class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        loss = 0
        iou_sum = 0
        for i in range(len(outputs)):
            outx, outy, out_size = tuple(outputs[i])
            tarx, tary, tar_size = tuple(targets[i])
            outputs_area = (2*out_size)**2
            targets_area = (2*tar_size)**2
            right_out, left_out = outx + out_size, outx-out_size
            right_tar, left_tar = tarx+tar_size, tarx-tar_size
            bottom_out, top_out = outy + out_size, outy-out_size
            bottom_tar, top_tar = tary+tar_size, tary-tar_size
            w, h = 0, 0
            if right_tar <= left_out:
                w = 0
            elif right_out <= left_tar:
                w = 0
            else:
                _, x1, x2, _ = tuple(
                    sorted([right_out, left_out, right_tar, left_tar]))
                w = x2-x1
            if top_tar >= bottom_out:
                h = 0
            elif top_out >= bottom_tar:
                h = 0
            else:
                _, y1, y2, _ = tuple(
                    sorted([bottom_out, top_out, bottom_tar, top_tar]))
                h = y2-y1
            overlap_area = w*h
            union = outputs_area+targets_area - overlap_area
            iou = overlap_area/(union+1e-7)
            iou_sum += iou
            loss = loss - torch.log(iou+1e-7)
        return loss, iou_sum
