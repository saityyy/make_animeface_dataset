from torch import nn
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
