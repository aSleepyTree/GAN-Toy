import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 设备配置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 超参数配置
config = {
    "latent_dim": 32,
    "data_dim": 2,
    "batch_size": 1280,
    "epochs": 500000,
    "lr": 0.0002,
    "d_g_ratio": 5,               # 判别器:生成器训练次数比例
    "checkpoint_dir": "./gan_checkpoints",
    "sample_dir": "./gan_samples", # 新增样本保存目录
    "save_interval": 1000,          # 保存间隔epoch数
    "num_test_samples": 5000,     # 新增测试样本数量
}

# 创建目录
os.makedirs(config["checkpoint_dir"], exist_ok=True)
os.makedirs(config["sample_dir"], exist_ok=True)  # 新增样本目录

# 真实数据生成函数（保持原样）
def generate_real_data(n_samples):
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = np.random.normal(0, 0.001, n_samples)
    x = np.stack([theta, np.sin(theta)], axis=1) + r[:, None]
    x[:, 0] = (x[:, 0] - np.pi) / np.pi
    return torch.FloatTensor(x).to(device)

# 网络结构定义（保持原样）
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config["latent_dim"], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, config["data_dim"]),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config["data_dim"], 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# 新增可视化保存函数
def save_samples(generator, epoch, show=False):
    with torch.no_grad():
        # 生成测试数据
        z = torch.randn(config["num_test_samples"], config["latent_dim"]).to(device)
        generated_data = generator(z).cpu().numpy()
        
        # 反归一化x轴
        generated_data[:, 0] = (generated_data[:, 0] + 1) * np.pi
        
        # 创建画布
        plt.figure(figsize=(10, 6))
        plt.scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.5)
        plt.title(f"Generated Samples at Epoch {epoch}")
        plt.xlim(0, 2*np.pi)
        plt.ylim(-1.5, 1.5)
        
        # 保存文件
        filename = f"{config['sample_dir']}/epoch_{epoch:04d}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        if show:
            plt.show()
        plt.close()
        
        print(f"Saved samples to {filename}")

# 模型保存/加载函数（保持原样）
def save_models(epoch):
    torch.save({
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "g_optimizer": g_optimizer.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "epoch": epoch
    }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch}.pth")
    print(f"Saved checkpoint at epoch {epoch}")

def load_models(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer"])
    return checkpoint["epoch"]

# 初始化模型和优化器（保持原样）
generator = Generator().to(device)
discriminator = Discriminator().to(device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=config["lr"])
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config["lr"])
loss_fn = nn.BCELoss()
temp = load_models('./gan_checkpoints/checkpoint_epoch_15000.pth')
# 训练循环（添加样本生成）
for epoch in range(temp,config["epochs"]):
    # 判别器训练（保持原样）
    for _ in range(config["d_g_ratio"]):
        d_optimizer.zero_grad()
        real_data = generate_real_data(config["batch_size"])
        real_labels = torch.ones(config["batch_size"], 1).to(device)
        d_real_loss = loss_fn(discriminator(real_data), real_labels)
        
        z = torch.randn(config["batch_size"], config["latent_dim"]).to(device)
        fake_data = generator(z).detach()
        fake_labels = torch.zeros(config["batch_size"], 1).to(device)
        d_fake_loss = loss_fn(discriminator(fake_data), fake_labels)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

    # 生成器训练（保持原样）
    g_optimizer.zero_grad()
    z = torch.randn(config["batch_size"], config["latent_dim"]).to(device)
    fake_data = generator(z)
    g_loss = loss_fn(discriminator(fake_data), real_labels)
    g_loss.backward()
    g_optimizer.step()

    # 定期保存和可视化
    if (epoch + 1) % config["save_interval"] == 0:
        save_models(epoch + 1)
        save_samples(generator, epoch + 1)  # 新增样本生成
        
    # 打印日志（保持原样）
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# 最终保存
save_models(config["epochs"])
save_samples(generator, config["epochs"])