import torch
from torch import nn


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
