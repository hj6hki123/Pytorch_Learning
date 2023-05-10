# 定義模型

import torch.nn as nn

num_classes = 50  # 50音分類
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(  # <-- (1,32,32)
            nn.Conv2d(1, 16, kernel_size=3, padding=1), #--> (32,32,32)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) #--> (16,16,16)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),#--> (32,16,16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) #--> (32,8,8)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),#--> (64,8,8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))#--> (64,4,4)
        self.fc1 = nn.Sequential(
            nn.Linear(4*4*64, 256), #--> (256)
            nn.Dropout(0.2),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Linear(256,num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out