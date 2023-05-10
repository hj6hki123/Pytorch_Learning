# 定義模型

import torch.nn as nn

num_classes = 50  # 50音分類
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(  # <-- (1,32,32)
            nn.Conv2d(1, 16, kernel_size=6,stride=2, padding=3), # --> (16,16,16)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))  # --> (16,8,8)
        self.fc = nn.Linear(16*8*8,num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out