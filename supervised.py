import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models


# 定义线性分类器
class LinearClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(LinearClassifier, self).__init__()
        self.encoder = pretrained_model
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

        # 在每个epoch结束时评估测试集上的准确率
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
            torch.save(linear_model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return best_accuracy


# 创建数据加载器
def create_data_loaders(batch_size):
    cifar100_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar100_transform)
    cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar100_transform)

    train_loader_cifar100 = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_cifar100 = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader_cifar100, test_loader_cifar100


if __name__ == '__main__':

    learning_rates = [5e-4]
    batch_sizes = [ 128]

    for lr in learning_rates:
        for bs in batch_sizes:

            train_loader_cifar100, test_loader_cifar100 = create_data_loaders(batch_size=bs)


            pretrained_model = models.resnet18(pretrained=True)
            linear_model = LinearClassifier(pretrained_model, num_classes=100)
            writer = SummaryWriter(log_dir=f'runs/supervised_lr{lr}_bs{bs}')


            best_accuracy = evaluate_linear_classifier(linear_model, train_loader_cifar100, test_loader_cifar100,
                                                       epochs=50, learning_rate=lr, writer=writer,
                                                       model_path=f'supervised_cifar100_resnet18_lr{lr}_bs{bs}.pth')
            print(f'Best Test Accuracy: {best_accuracy:.2f}%')
            writer.close()
