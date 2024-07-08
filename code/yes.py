import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 100)  # 第一个全连接层
        self.fc2 = nn.Linear(100, 10)   # 第二个全连接层

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = F.softmax(x, dim=1)
        return x
    

# 初始化模型
model = MyModel()

# 加载模型参数
model.load_state_dict(torch.load('../param/myFirstModel.pth'))

# 将模型设置为评估模式
model.eval()

print('模型参数已加载并设置为评估模式')

# 定义转换操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 测试模型
def test(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # 累积损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取预测标签
            correct += pred.eq(target.view_as(pred)).sum().item()  # 计算正确数量

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

# 对数据进行测试
test(model, test_loader)