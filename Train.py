import copy
import os
import time
from datetime import datetime

import pandas as pd
import torch
import torch.utils.data as Data
from matplotlib import pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms

from AlexNet import AlexNet


def train_loader():
    dataset = datasets.CIFAR10(root='./data',
                               train=True,
                               download=True,
                               transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]))
    train_data, valid_data = Data.random_split(dataset, [round(len(dataset) * 0.8), round(len(dataset) * 0.2)])
    train_data_loader, valid_data_loader = (
        Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=2),
        Data.DataLoader(dataset=valid_data, batch_size=64, shuffle=True, num_workers=2))
    print("Data Loaded.")
    print("{}".format(dataset.classes))
    return train_data_loader, valid_data_loader


def train_model(model, epochs, train_data_loader, valid_data_loader, criterion=nn.CrossEntropyLoss(),
                Optimizer=optim.SGD,
                learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_on_device = model.to(device)
    optimizer = Optimizer(net_on_device.parameters(), lr=learning_rate, momentum=0.9)

    best_valid_accuracies = 0
    train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []
    train_loss, valid_loss, train_acc, valid_acc = 0.0, 0.0, 0.0, 0.0
    train_num, valid_num = 0, 0

    for epoch in range(epochs):
        since = time.time()
        print('Epoch: {}/{}'.format(epoch, epochs - 1))
        print('-' * 15)
        for batch_idx, (data, target) in enumerate(train_data_loader):
            images, labels = data.to(device), target.to(device)
            net_on_device.train()
            output = net_on_device(images).double()
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, labels)  # todo Creterion Error!
            train_acc += torch.sum(pre_lab == labels).item()
            train_loss += loss.item() * images.size(0)
            train_num += images.size(0)
            net_on_device.zero_grad()

            # 模型更新, 反向传播
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_data_loader):
                images, labels = data.to(device), target.to(device)
                net_on_device.eval()
                output = net_on_device(images).double()
                pre_lab = torch.argmax(output, dim=1)
                loss = criterion(output, labels)
                valid_acc += torch.sum(pre_lab == labels).item()
                valid_loss += loss.item() * images.size(0)
                valid_num += images.size(0)

        train_losses.append(train_loss / train_num)
        train_accuracies.append(train_acc / train_num)
        valid_losses.append(valid_loss / valid_num)
        valid_accuracies.append(valid_acc / valid_num)
        print('Train Loss : \t\tTrain acc \t-----> {:.5f} : \t\t{:.5f}'
              .format(train_losses[-1], train_accuracies[-1]))
        print('Val Loss : \t\t\tVal acc \t-----> {:.5f} : \t\t\t{:.5f}'
              .format(valid_losses[-1], valid_accuracies[-1]))

        if valid_accuracies[-1] > best_valid_accuracies:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_valid_accuracies = valid_accuracies[-1]
            # 保存模型的操作移到这里，减少文件I/O
            model_dir = './model'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            torch.save(best_model_wts, model_path)

        time_elapsed = time.time() - since
        print('Training complete in {:.1f}s'.format(time_elapsed % 60))
    train_process = pd.DataFrame(data={"epoch": range(epochs),
                                       "train_loss": train_losses,
                                       "train_acc": train_accuracies,
                                       "val_loss": valid_losses,
                                       "val_acc": valid_accuracies,
                                       })
    return train_process


def plot_loss_and_acc(dataframe):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(dataframe['epoch'], dataframe['train_loss'], 'ro-', label="Training Loss")
    plt.plot(dataframe['epoch'], dataframe['val_loss'], 'bs-', label="Validation Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(dataframe['epoch'], dataframe['train_acc'], 'ro-', label="Training Accuracy")
    plt.plot(dataframe['epoch'], dataframe['val_acc'], 'bs-', label="Validation Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    model = AlexNet()
    model.load_state_dict(torch.load('./model/best_model.pth'))
    train_loader, val_loader = train_loader()
    train_process = train_model(model=model,
                                train_data_loader=train_loader,
                                valid_data_loader=val_loader,
                                epochs=1,
                                learning_rate=0.01,
                                Optimizer=optim.SGD)
    plot_loss_and_acc(train_process)
    counter = 1
    path = './result/'
    filename = "train_process.csv"  # 初始文件名应该包括.csv
    full_path = os.path.join(path, filename)  # 使用os.path.join来构建完整路径

    # 检查目录是否存在，如果不存在，则创建
    if not os.path.exists(path):
        os.makedirs(path)

    # 检查文件是否存在，如果存在，则更新文件名
    while os.path.exists(full_path):
        new_filename = f"train_process({counter}).csv"  # 更新文件名，保证.csv在末尾
        full_path = os.path.join(path, new_filename)  # 更新完整路径
        counter += 1
    train_process.to_csv(full_path, index=False)
