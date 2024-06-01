import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models



class ResNet18WithDropout(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18WithDropout, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def train_supervised(model, train_loader, test_loader, epochs, learning_rate, writer, model_path, patience=10):
    model = model.cuda()  # 确保模型在GPU上
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()  # 确保输入数据和标签在GPU上
            outputs = model(inputs)
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
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # 在每个epoch结束时评估测试集上的准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()  # 确保输入数据和标签在GPU上
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_accuracy = 100. * correct / total
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        print(f"Test Accuracy: {test_accuracy:.2f}%")

        # 早停机制
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), model_path)  # 保存最佳模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return best_accuracy


# 创建数据加载器
def create_data_loaders(batch_size):
    cifar100_transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    cifar100_transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar100_transform_train)
    cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar100_transform_test)

    train_loader_cifar100 = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_cifar100 = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader_cifar100, test_loader_cifar100


if __name__ == '__main__':
    # 运行实验
    learning_rate = 1e-3
    batch_size = 128
    epochs = 50
    patience = 10

    # 创建数据加载器
    train_loader_cifar100, test_loader_cifar100 = create_data_loaders(batch_size=batch_size)

    # 初始化带有dropout的ResNet-18模型
    model = ResNet18WithDropout(num_classes=100)
    writer = SummaryWriter(log_dir='runs/supervised_cifar100')

    # 训练模型
    best_accuracy = train_supervised(model, train_loader_cifar100, test_loader_cifar100, epochs=epochs,
                                     learning_rate=learning_rate, writer=writer,
                                     model_path='supervised_cifar100_resnet18.pth', patience=patience)
    print(f'Best Test Accuracy: {best_accuracy:.2f}%')
    writer.close()
