import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

#有问题！但是python学的不够，不会改，跑出来的速度超级慢，有所借鉴他人代码，附在此处：https://blog.csdn.net/m0_46166352/article/details/114272172

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义全连接神经网络模型,和question01一样
class SimpleFCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义评估指标函数
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():#所有计算得出的t自动设置为False
        for images, labels in dataloader:
            outputs = model(images.view(images.size(0), -1))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# 设置超参数
input_size = 28 * 28
hidden_size = 128
output_size = 10
num_epochs = 10
learning_rate = 0.001

# 初始化模型、损失函数和优化器
model = SimpleFCNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
train_losses = []
test_accuracies = []
for epoch in range(num_epochs):#10
    model.train()#保证BN层能够用到每一批数据的均值和方差。
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images.view(images.size(0), -1))#展开
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    test_accuracy = evaluate_model(model, test_loader)
    test_accuracies.append(test_accuracy)
    print(f'Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy}')

# 绘制训练损失和测试准确率曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)#生成一行两列两个子图
plt.plot(train_losses)
plt.title('Training Loss')#损失
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)#生成一行两列两个子图
plt.plot(test_accuracies)
plt.title('Test Accuracy')#准确
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()
