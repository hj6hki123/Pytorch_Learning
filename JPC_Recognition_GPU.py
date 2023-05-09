import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# 檢查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超參數
num_classes = 50  # 50音分類
batch_size = 20  # 訓練樣本數
num_epochs = 30  # 一期訓練  Iteration = （Data set size / Batch size）* Epoch = (1000/20)*10 = 5000
learning_rate = 0.001  # 學習率

# 數據預處理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 調整大小為 32x32
    transforms.Grayscale(num_output_channels=1),  # 轉為單通道灰度圖像
    transforms.ToTensor()  # 轉為張量
])

# 載入數據 並將數據分配測試集和訓練集 8:2
train_dataset = datasets.ImageFolder(root='./my_images', transform=transform)  # datasets.ImageFolder 會將資料夾中的圖片依照資料夾名稱分類
train_data, test_data = train_test_split(train_dataset, test_size=0.2,
                                         random_state=42)  # train_test_split 會將資料分成兩部分，一部分用來訓練，一部分用來測試
# 訓練集
train_loader = DataLoader(train_data, batch_size=batch_size,
                          shuffle=True)  # 每次訓練數量 = Data set size(*0.8) / Batch size = 800/20 = 40
# 測試集
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# 定義模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 50)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


# 實例化模型、損失函數和優化器
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 可視化模型
from torchsummary import summary

summary(model, input_size=(1, 32, 32), device='cuda')


# 訓練模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], \
                    Step [{i + 1}/{total_step}], \
                    Loss: {loss.item():.4f}')




# 測試模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))


# 保存模型
torch.save(model.state_dict(), '50on_model.ckpt')

from PIL import Image

# 轉換圖片
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 調整大小為 32x32
    transforms.Grayscale(num_output_channels=1),  # 轉為單通道灰度圖像
    transforms.ToTensor()  # 轉為張量
])

# 載入圖片
img = Image.open('./test_images/HE.jpg')
img = test_transform(img)

# 增加一維，批次大小設為 1
img = img.unsqueeze(0)

# 使用訓練好的模型進行預測
img = img.to(device)
model = model.to(device)
with torch.no_grad():
    output = model(img)
    pred = output.argmax(dim=1)

label_dict = train_dataset.class_to_idx
reverse_label_dict = {v: k for k, v in label_dict.items()}
print(f'預測結果:{reverse_label_dict[pred.item()]}')
