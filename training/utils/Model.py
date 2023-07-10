from torch import nn


class Model(nn.Module):
    def __init__(self, m):
        super(Model, self).__init__()
        self.loss = []
        self.acc = []
        self.resnet = m
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
