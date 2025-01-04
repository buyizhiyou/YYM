import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

# 设置设备为 CUDA:1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 定义 RealNVP 模型
class RealNVP(nn.Module):
    def __init__(self, num_features, hidden_dim=512, num_layers=8):
        super(RealNVP, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 每一层的 scale (s) 和 translation (t) 网络
        self.s_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(num_features // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features // 2)
        ) for _ in range(num_layers)])

        self.t_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(num_features // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features // 2)
        ) for _ in range(num_layers)])

    def forward(self, x, reverse=False):
        """
        前向或逆向变换。
        如果 reverse=True，则执行逆变换；否则执行正向变换。
        """
        x1, x2 = x.chunk(2, dim=1)  # 将输入分为两部分
        for i in range(self.num_layers):
            s = self.s_layers[i](x1)  # 缩放因子
            t = self.t_layers[i](x1)  # 平移因子
            if reverse:  # 逆变换
                x2 = (x2 - t) * torch.exp(-s)
            else:  # 正向变换
                x2 = x2 * torch.exp(s) + t
            x1, x2 = x2, x1  # 每层交换 x1 和 x2
        
        return torch.cat([x1, x2], dim=1)

    def log_prob(self, x):
        """
        计算输入 x 的对数概率密度。
        """
        x1, x2 = x.chunk(2, dim=1)
        log_det = 0  # 初始化对数雅可比行列式
        for i in range(self.num_layers):
            s = self.s_layers[i](x1)
            t = self.t_layers[i](x1)
            log_det += torch.sum(s, dim=1)  # 累积对数雅可比行列式
            x2 = (x2 - t) * torch.exp(-s)  # 更新 x2
            x1, x2 = x2, x1  # 每层交换 x1 和 x2

        z = torch.cat([x1, x2], dim=1)  # 最终的 z
        # 标准正态分布的对数概率密度
        log_prob_z = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * self.num_features * torch.log(torch.tensor(2 * torch.pi))
        return log_prob_z + log_det  # 综合对数概率

# 生成数据集
def generate_data(num_samples, num_features=2048):
    """
    生成正态分布的数据集。
    """
    return torch.randn(num_samples, num_features).to(device)

# 实例化模型并移到设备上
num_features = 2048
model = RealNVP(num_features).to(device)

# 生成训练集和测试集
train_data = generate_data(50000, num_features)
test_data = generate_data(1000, num_features)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        # 数据的总数量
        return len(self.data)

    def __getitem__(self, idx):
        # 返回指定索引的数据
        return self.data[idx]

# 将数据加载到自定义的 Dataset 中
dataset = CustomDataset(train_data)
# 创建 DataLoader，配置 batch_size 为 256
batch_size = 1024
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 使用 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 归一化???
# train_data = (train_data - train_data.mean(dim=0)) / train_data.std(dim=0)
# test_data = (test_data - train_data.mean(dim=0)) / train_data.std(dim=0)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    for x in tqdm(dataloader):
        model.train()
        optimizer.zero_grad()

        # 计算负对数似然作为损失
        loss = -torch.mean(model.log_prob(train_data))
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 测试阶段计算测试集的概率密度
model.eval()
with torch.no_grad():
    log_prob_test = model.log_prob(test_data)  # 计算测试集对数概率
    prob_density_test = torch.exp(log_prob_test)  # 概率密度

# 打印部分测试集的概率密度结果
print("Test set probability densities (first 10):")
print(prob_density_test[:10])
