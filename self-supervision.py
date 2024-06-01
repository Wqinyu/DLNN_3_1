import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision.models as models


# 定义BYOL模型
class BYOL(nn.Module):
    def __init__(self, base_model, out_dim=512):
        super(BYOL, self).__init__()
        self.online_encoder = base_model(pretrained=False)
        self.online_encoder.fc = nn.Identity()
        self.predictor = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, out_dim)
        )
        self.target_encoder = base_model(pretrained=False)
        self.target_encoder.fc = nn.Identity()
        self._update_target_encoder(0)

    def forward(self, x):
        online_proj = self.predictor(self.online_encoder(x))
        with torch.no_grad():
            target_proj = self.target_encoder(x)
        return online_proj, target_proj

    def _update_target_encoder(self, m):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = m * target_params.data + (1 - m) * online_params.data


def byol_loss(out_1, out_2):
    return 2 - 2 * F.cosine_similarity(out_1, out_2).mean()


def train_byol(model, train_loader, epochs, learning_rate, target_update_rate, writer, model_path, patience=10):
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.cuda()
            online_proj, target_proj = model(images)
            loss = byol_loss(online_proj, target_proj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            model._update_target_encoder(target_update_rate)
        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)  # 保存最佳模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


# 定义线性分类器
class LinearClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(LinearClassifier, self).__init__()
        self.encoder = base_model(pretrained=False)
        self.encoder.fc = nn.Identity()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.fc(features)


def evaluate_linear_classifier(linear_model, train_loader, test_loader, epochs, learning_rate, writer, model_path,
                               patience=10):
    linear_model = linear_model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_model.fc.parameters(), lr=learning_rate)
    best_accuracy = 0
    patience_counter = 0

    for epoch in range(epochs):
        linear_model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = linear_model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        writer.add_scalar('Loss/linear_train', avg_loss, epoch)
        writer.add_scalar('Accuracy/linear_train', accuracy, epoch)
        print(f"Epoch [{epoch + 1}/20], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # 在每个epoch结束时评估验证集上的准确率
        linear_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = linear_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_accuracy = 100. * correct / total
        writer.add_scalar('Accuracy/linear_test', test_accuracy, epoch)
        print(f"Test Accuracy: {test_accuracy:.2f}%")

        # 早停机制
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            torch.save(linear_model.state_dict(), model_path)  # 保存最佳模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return best_accuracy


# 创建数据加载器
def create_data_loaders(subset_ratio, batch_size):
    svhn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    svhn_train = datasets.SVHN(root='./data', split='train', download=True, transform=svhn_transform)
    if subset_ratio < 1.0:
        indices = list(range(len(svhn_train)))
        np.random.shuffle(indices)
        subset_indices = indices[:int(len(indices) * subset_ratio)]
        svhn_train = Subset(svhn_train, subset_indices)
    train_loader = DataLoader(svhn_train, batch_size=batch_size, shuffle=True, num_workers=4)

    cifar100_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar100_transform)
    cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar100_transform)

    train_loader_cifar100 = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_cifar100 = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, train_loader_cifar100, test_loader_cifar100


if __name__ == '__main__':
    # 运行实验
    learning_rates = [5e-4, 1e-4]
    batch_sizes = [256, 128]
    subset_ratios = [1.0, 0.5]

    for lr in learning_rates:
        for bs in batch_sizes:
            for sr in subset_ratios:
                # 创建数据加载器
                train_loader, train_loader_cifar100, test_loader_cifar100 = create_data_loaders(subset_ratio=sr,
                                                                                                batch_size=bs)

                # 训练BYOL模型
                model = BYOL(models.resnet18, out_dim=512)
                writer = SummaryWriter(log_dir=f'runs/byol_lr{lr}_bs{bs}_sr{sr}')
                train_byol(model, train_loader, epochs=100, learning_rate=lr, target_update_rate=0.996, writer=writer,
                           model_path=f'byol_svhn_resnet18_lr{lr}_bs{bs}_sr{sr}.pth', patience=5)
                writer.close()

                # 评估线性分类器
                linear_model = LinearClassifier(models.resnet18, num_classes=100)
                linear_model.encoder.load_state_dict(model.online_encoder.state_dict())
                writer = SummaryWriter(log_dir=f'runs/linear_lr{lr}_bs{bs}_sr{sr}')
                best_accuracy = evaluate_linear_classifier(linear_model, train_loader_cifar100, test_loader_cifar100,
                                                           epochs=50, learning_rate=lr, writer=writer,
                                                           model_path=f'linear_cifar100_resnet18_lr{lr}_bs{bs}_sr{sr}.pth',
                                                           patience=7)
                print(f'Best Test Accuracy: {best_accuracy:.2f}%')
                writer.close()
