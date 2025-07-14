import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import random_split
from tqdm import tqdm
from datasets import load_from_disk
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
from datasets import load_dataset
from torchvision import models

#设置可以使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

class train(nn.Module):
    def __init__(self):
        super(train, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            MaxPool2d(2),
            
            Conv2d(32, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            MaxPool2d(2, stride=2),
            
            Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            MaxPool2d(2, stride=2),
            
            Flatten(),
            Linear(16384, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            Linear(1024, 200),
        )
    def forward(self,x):
        x = self.model(x)
        return x

class PretrainedModel(nn.Module):
    def __init__(self, num_classes=200):
        super(PretrainedModel, self).__init__()
        # 加载预训练的ResNet18
        from torchvision.models import ResNet18_Weights
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 或使用 DEFAULT
        
        # 冻结大部分层的参数，只训练最后几层
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
            
        # 替换最后的全连接层以适应你的任务
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class dataset_deduplication(Dataset):
    def __init__(self, path, transform=None):
        self.dataset = load_from_disk(path)
        self.transform = transform
        self.imgs = self.dataset["image"]
        self.labels = self.dataset["label"]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        image = self.imgs[idx].convert("RGB")  # 直接使用已加载的图像对象#强制转换成RGB格式三通道
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

class dataset_undeduplication(Dataset):
    def __init__(self, path, transform=None):
        self.dataset = load_dataset('Maysee/tiny-imagenet', split='train')
        self.transform = transform
        self.imgs = self.dataset["image"]
        self.labels = self.dataset["label"]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        image = self.imgs[idx].convert("RGB")#强制转换成RGB格式三通道
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

class dataset_undeduplication_valid(Dataset):
    def __init__(self, path, transform=None):
        self.dataset = load_dataset('Maysee/tiny-imagenet', split='valid')
        self.transform = transform
        self.imgs = self.dataset["image"]
        self.labels = self.dataset["label"]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        image = self.imgs[idx].convert("RGB")#强制转换成RGB格式三通道
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
    
#dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # ResNet期望224×224输入
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准化参数
])

full_dataset_deduplication = dataset_deduplication("D:/桌面/Project-A/the_final_result/ncentroids_2000/dataset", transform=transform)#这里需要注意一下\是转义符号的意思，需要用/。
full_dataset = dataset_undeduplication("D:/桌面/Project-A/dataset", transform=transform)
test_dataset = dataset_undeduplication_valid("D:/桌面/Project-A/dataset_test", transform=transform)

def run_training(dataset_name, full_dataset, test_dataset, epochs=30, patience=5):
    """使用给定数据集运行完整的训练流程并返回准确率历史"""
    
    # 分割数据集

    train_size = len(full_dataset)
    test_size = len(test_dataset)
    train_dataset = full_dataset
    test_dataset = test_dataset
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型和优化器
    model = PretrainedModel(num_classes=200).to(device)
    # 增加权重衰减，改善泛化能力
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 训练记录
    train_accuracies = []
    test_accuracies = []
    train_losses = []    # 新增：记录训练loss
    test_losses = []     # 新增：记录测试loss
    
    # 早停相关变量
    best_accuracy = 0.0
    best_model_state = None
    patience_counter = 0
    
    # 训练循环
    for t in range(epochs):
        print(f"[{dataset_name}] Epoch {t+1}/{epochs}")
        
        # 训练
        model.train()
        correct_train = 0
        total_samples = len(train_loader.dataset)
        running_train_loss = 0.0  # 新增：记录每个epoch的总训练loss
        
        for images, labels in tqdm(train_loader, desc=f"{dataset_name} epoch", leave=False):
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            loss = loss_fn(pred, labels)
            correct_train += (pred.argmax(1) == labels).type(torch.float).sum().item()
            running_train_loss += loss.item() * images.size(0)  # 新增：累加loss乘以batch大小 .item()是获得int或者float类型的值
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 计算训练准确率和平均loss
        train_accuracy = correct_train / total_samples
        train_accuracies.append(train_accuracy)
        epoch_train_loss = running_train_loss / total_samples  # 新增：计算平均训练loss
        train_losses.append(epoch_train_loss)  # 新增：记录平均训练loss
        
        # 测试
        model.eval()
        test_loss, correct_test = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images)
                test_loss += loss_fn(pred, labels).item() * images.size(0)  # 修改：累加loss乘以batch大小
                correct_test += (pred.argmax(1) == labels).type(torch.float).sum().item()
        
        # 计算测试准确率和平均loss
        test_accuracy = correct_test / len(test_loader.dataset)
        test_accuracies.append(test_accuracy)
        epoch_test_loss = test_loss / len(test_loader.dataset)  # 新增：计算平均测试loss
        test_losses.append(epoch_test_loss)  # 新增：记录平均测试loss
        
        # 早停检查
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  新的最佳测试准确率: {(100*best_accuracy):>0.1f}%")
        else:
            patience_counter += 1
            print(f"  未改善，耐心值: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"  已经连续 {patience} 轮没有改善，提前终止训练")
            break
        
        # 输出当前结果，增加loss信息
        print(f"  训练准确率: {(100*train_accuracy):>0.1f}%, 测试准确率: {(100*test_accuracy):>0.1f}%")
        print(f"  训练Loss: {epoch_train_loss:.4f}, 测试Loss: {epoch_test_loss:.4f}")
        
        # 更新学习率
        scheduler.step(test_accuracy)
    
    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    
    # 最终测试结果
    print(f"\n[{dataset_name}] 最佳测试准确率: {(100*best_accuracy):>0.1f}%")
    
    # 保存最佳模型
    torch.save(best_model_state, f"{dataset_name}_best_model.pth")
    
    return {
        'train_acc': train_accuracies,
        'test_acc': test_accuracies,
        'train_loss': train_losses,  # 新增：返回训练loss
        'test_loss': test_losses,    # 新增：返回测试loss
        'best_acc': best_accuracy,
        'name': dataset_name
    }

deduped_results = run_training("deduped", full_dataset_deduplication, test_dataset, epochs=25, patience=5)
undeduped_results = run_training("undeduped", full_dataset, test_dataset, epochs=25, patience=5)

# 计算实际训练的轮数
actual_undedup_epochs = len(undeduped_results['train_acc'])
actual_dedup_epochs = len(deduped_results['train_acc'])

# 绘制训练准确率和loss对比图
plt.figure(figsize=(16, 12))  # 增大图表尺寸以容纳更多子图

# 子图1: 训练准确率对比
plt.subplot(2, 2, 1)
plt.plot(range(1, actual_undedup_epochs+1), undeduped_results['train_acc'], 'b-', label='undeduped-accuracy')
plt.plot(range(1, actual_dedup_epochs+1), deduped_results['train_acc'], 'r-', label='deduped-accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.title('train-accuracy')
plt.legend()
plt.grid(True)
51.7
# 子图2: 测试准确率对比
plt.subplot(2, 2, 2)
plt.plot(range(1, actual_undedup_epochs+1), undeduped_results['test_acc'], 'b-', label='undeduped-accuracy')
plt.plot(range(1, actual_dedup_epochs+1), deduped_results['test_acc'], 'r-', label='deduped-accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.title('test-accuracy')
plt.legend()
plt.grid(True)

# 子图3: 训练Loss对比
plt.subplot(2, 2, 3)
plt.plot(range(1, actual_undedup_epochs+1), undeduped_results['train_loss'], 'b-', label='undeduped-Loss')
plt.plot(range(1, actual_dedup_epochs+1), deduped_results['train_loss'], 'r-', label='deduped-Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('train-Loss')
plt.legend()
plt.grid(True)

# 子图4: 测试Loss对比
plt.subplot(2, 2, 4)
plt.plot(range(1, actual_undedup_epochs+1), undeduped_results['test_loss'], 'b-', label='undeduped-test-Loss')
plt.plot(range(1, actual_dedup_epochs+1), deduped_results['test_loss'], 'r-', label='deduped-test-Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('test-Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('accuracy_and_loss_comparison.png', dpi=300)
plt.show()
# 保存模型
# torch.save(model.state_dict(), "cat_and_dog_model.pth")
# writer.close()