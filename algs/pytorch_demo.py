import json
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

with open("c:\\Users\\fanbin\\Desktop\\auth\\agent\\algs\\labels.json") as f:
    labels_data = json.load(f)

# 加载预训练的ResNet
# 自动下载 https://download.pytorch.org/models/resnet18-f37072fd.pth
model = models.resnet18(pretrained=True)
# 本地模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.load("/path/to/model.pth", map_location=device)
model.eval()  # 设置为评估模式

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载并预处理图像
img = Image.open("c:\\Users\\fanbin\\Desktop\\static\\msn.jpeg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
# 加载训练集

# 预测
out = model(batch_t)
_, index = torch.max(out, 1)  # 获取的索引是该向量中最大值的位置，即模型预测概率最高的类别编号
print("预测类别索引: ", index.item())  # 例如值是525，表示模型认为该输入图像属于第525类
print("预测类别是: ", labels_data[index.item()])  # dam


# 加载训练集
train_dataset = datasets.ImageFolder(
    root="./path/",
    transform=transform
)
# 加载验证集
val_dataset = datasets.ImageFolder(
    root="./path/val",
    transform=transform
)
class_names = train_dataset.classes
class_to_idx = train_dataset.class_to_idx
print(f"类别数量: ", len(class_names))
print(f"类别映射: ", class_to_idx)

batch_size = 32
num_workers = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 神经网络示例
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建数据集合数据加载起
data = torch.randn(1000, 10)  # 1000个样本，每个10维
labels = torch.randint(0, 2, (1000,))  # 1000个二元标签

dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 使用数据加载器
for batch_data, batch_labels in dataloader:
    print("批次数据: ", batch_data.shape)
    print("批次标签: ", batch_labels.shape)
    break


# # 创建网络、损失函数和优化器
# net = SimpleNet()
# criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01)

# inputs = torch.randn(100, 10)
# targets = torch.randn(100, 1)
# for epoch in range(100):
#     # 前向传播
#     outputs = net(inputs)
#     loss = criterion(outputs, targets)  # 损失
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if epoch % 10 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item()}')


# # 创建张量
# x = torch.tensor([1, 2, 3])
# y = torch.tensor([4, 5, 6])

# z = x + y
# print("加法: ", z)

# # 矩阵乘法
# a = torch.randn(2, 3)
# b = torch.randn(3, 2)

# if torch.cuda.is_available():
#     x = x.cuda()
#     y = y.cuda()
#     z = x + y
#     print("Gpu上的加法: ", z)


# # 自动求导
# x = torch.tensor(2.0, requires_grad=True)
# y = x**2 + 3*x + 1
# y.backward()
# print("x=2时，y关于x的梯度: ", x.grad)
