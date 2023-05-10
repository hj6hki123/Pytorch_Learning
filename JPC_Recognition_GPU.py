#!/usr/bin/env python
# coding: utf-8

# # 圖片分類

# In[119]:


import os
import shutil

# 遍歷資料夾中的檔案
for filename in os.listdir('./hiragana_images'):
    # 判斷檔案是否為 jpg 檔
    if filename.endswith('.jpg'):
        # 取得檔名中的非數字字元
        class_name = os.path.splitext(''.join(filter(lambda x: not x.isdigit(), filename)))[0]
        # 建立目標資料夾
        target_folder = os.path.join('./my_images', class_name)
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        # 複製檔案到對應的資料夾中
        shutil.copy(os.path.join('./hiragana_images', filename), os.path.join(target_folder, filename))

# # 模型訓練

# In[120]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# In[121]:


# 檢查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# In[122]:


# 超參數

num_classes = 50  # 50音分類
batch_size = 10  # 訓練樣本數
num_epochs = 20  # 一期訓練  Iteration = （Data set size / Batch size）* Epoch = (1000/20)*10 = 5000
learning_rate = 0.001  # 學習率

# In[123]:


# 數據預處理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 調整大小為 32x32
    transforms.Grayscale(num_output_channels=1),  # 轉為單通道灰度圖像
    transforms.ToTensor()  # 轉為張量
])

# In[124]:


# 載入數據 並將數據分配測試集和訓練集 8:2
train_dataset = datasets.ImageFolder(root='./my_images', transform=transform)  # datasets.ImageFolder 會將資料夾中的圖片依照資料夾名稱分類
train_data, test_data = train_test_split(train_dataset, test_size=0.2,
                                         random_state=42)  # train_test_split 會將資料分成兩部分，一部分用來訓練，一部分用來測試
# 訓練集
train_loader = DataLoader(train_data, batch_size=batch_size,
                          shuffle=True)  # 每次訓練數量 = Data set size(*0.8) / Batch size = 800/20 = 40
# 測試集
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# # 載入數據
# train_dataset = datasets.ImageFolder(root='F:\pycharm_pro\pytorch_learn\my_images', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 卷積層尺寸變化 =
# $$
# W_{new} = {W_{o} - Kernelsize + (2 \times Padding) \over Stride} +1
# $$


from models.All_Conv import Net

# 實例化模型、損失函數和優化器
model = Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 可視化模型
from torchsummary import summary

summary(model, input_size=(1, 32, 32), device=device.type)

# In[128]:


# 訓練模型
total_step = len(train_loader)  # 1 epoch = 800/batch = 40
loss_list = []
for epoch in range(num_epochs):
    loss = 0
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
    loss_list.append(loss.item())

# In[129]:


# 測試模型

label_dict = train_dataset.class_to_idx
reverse_label_dict = {v: k for k, v in label_dict.items()}

# 導入 Matplotlib 庫
from matplotlib import pyplot as plt

# 繪製訓練過程中的損失(loss)值變化圖
plt.plot(loss_list)
plt.title('Train Loss')
plt.show()

# 設置模型為評估模式
model.eval()

# 在不計算梯度的情況下測試模型
with torch.no_grad():
    # 初始化準確率計數器
    correct = 0
    total = 0
    error = []
    # 循環處理測試集中的每一個樣本
    for images, labels in test_loader:
        # 將樣本數據和標籤移到 GPU 上進行計算
        images = images.to(device)
        labels = labels.to(device)
        # 通過神經網絡進行預測
        outputs = model(images)
        # 找到每個樣本的最大值及其對應的索引，即預測類別
        _, predicted = torch.max(outputs.data, 1)
        # 統計正確預測的數量
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # 判斷預測值和實際標籤是否一致，如果不一致就把它們添加到錯誤列表中

    # 計算並輸出準確率
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    print(f'total:{total} ,\tcorrect:{correct}')

# # 儲存模型

# In[130]:


# 只儲存權重
torch.save(model.state_dict(), '50on_model.ckpt')
# 儲存整個模型和權重
torch.save(model, '50on_model_all.pt')

# # 載入模型

# In[134]:


import torch
from torchvision import transforms
from PIL import Image

# 定義圖片轉換
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 調整大小為 32x32
    transforms.Grayscale(num_output_channels=1),  # 轉為單通道灰度圖像
    transforms.ToTensor()  # 轉為張量

])
# 載入圖片
img = Image.open('./test_images/HE.jpg')

# 將圖片轉換為張量
img = test_transform(img).to(device)
# 增加一維，batch大小設為 1
img = img.unsqueeze(0)

# 載入模型
model = torch.load('50on_model_all.pt')
# 設置模型為評估模式
model.eval().to(device)

# 使用模型進行預測
with torch.no_grad():
    output = model(img)

# 取得預測結果
_, preds = torch.max(output, 1)
label_dict = train_dataset.class_to_idx
reverse_label_dict = {v: k for k, v in label_dict.items()}
print(f'預測結果:{reverse_label_dict[preds.item()]}')

# In[ ]:
