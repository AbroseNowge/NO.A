import torch
import torch.nn as nn

# 定义一个简单的两层全连接神经网络
class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义输入维度、隐藏层维度和输出维度
input_size = 128
hidden_size = 256
output_size = 10  # 输出维度示例，这里设置为 10

# 创建一个两层全连接神经网络实例
model = TwoLayerFC(input_size, hidden_size, output_size)

# 打印模型结构
print(model)
