import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import csv

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        # (TODO) Design your CNN, it can only be less than 3 convolution layers
        super(CNN, self).__init__()
        self.convLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((10, 10)),
        )
        self.fullConnect = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*10*10, 64*5),
            nn.ReLU(),
            nn.Linear(64*5, 64),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # (TODO) Forward the model
        x = self.convLayer(x)
        x = self.fullConnect(x)
        return x

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    # (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress
    model.train()
    
    total_loss = 0.0
    num_batch = 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        y_hat = model(images)
        train_loss = criterion(y_hat, labels)
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.detach().item()
        num_batch += 1.0
    avg_loss = total_loss/num_batch
    return avg_loss

@torch.no_grad()
def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    model.eval()

    total_loss = 0.0
    total_predict_success = 0.0
    num_batch = 0.0
    num_sample = 0.0

    for images, labels in tqdm(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        y_hat = model(images)
        val_loss = criterion(y_hat, labels)
        total_loss += val_loss.item()
        num_batch += 1.0
        num_sample += images.size(0)
        

        result = (torch.argmax(y_hat, dim=1) == labels)
        total_predict_success += result.sum().item()

    accuracy = total_predict_success/num_sample
    avg_loss = total_loss/num_batch

    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    with open("CNN.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "prediction"])
        for images, base_name in tqdm(test_loader):
            images = images.to(device)
            y_hat = model(images)
            result = torch.argmax(y_hat, dim=1)
            for id, predict in zip(base_name, result):
                writer.writerow([id, predict.item()])
    print(f"Predictions saved to 'CNN.csv'")
    return