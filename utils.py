from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name
    
def load_train_dataset(path: str='data/train/')->Tuple[List, List]:
    # (TODO) Load training dataset from the given path, return images and labels
    images = []
    labels = []

    for img in os.listdir(path+"elephant"):
        images.append(path+"elephant/"+img)
        labels.append(0)
    
    for img in os.listdir(path+"jaguar"):
        images.append(path+"jaguar/"+img)
        labels.append(1)

    for img in os.listdir(path+"lion"):
        images.append(path+"lion/"+img)
        labels.append(2)

    for img in os.listdir(path+"parrot"):
        images.append(path+"parrot/"+img)
        labels.append(3)
    
    for img in os.listdir(path+"penguin"):
        images.append(path+"penguin/"+img)
        labels.append(4)

    return images, labels

def load_test_dataset(path: str='data/test/')->List:
    # (TODO) Load testing dataset from the given path, return images
    images = []
    for img in os.listdir(path):
        images.append(path+img)
    return images

def plot(train_losses: List, val_losses: List):
    # (TODO) Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    #        xlabel: 'Epoch', ylabel: 'Loss'
    epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.plot(epoch, train_losses, label = "Train Loss")
    plt.plot(epoch, val_losses, label = "Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss.png")
    print("Save the plot to 'loss.png'")
    return