import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)


        self.fc1 = nn.Linear(512 * 1 * 1, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.pool(F.relu(self.bn7(self.conv7(x))))  

        x = x.view(x.size(0), -1)  
        x = self.dropout(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.fc2(x)
        return x



transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    #(0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
])



train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

classes = train_set.classes
print('classes =',  classes)


def train_model():
    model = ConvNeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()

    WEIGHT_DECAY_SGD = 5e-4
    EPOCHS = 60

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=WEIGHT_DECAY_SGD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_accuracy = evaluate_model(model)
        scheduler.step()
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.3f}, Val Acc: {val_accuracy:.2f}%')
        torch.save(model.state_dict(), 'cifar10_cnn.pth')
    return model

def evaluate_model(model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def evaluate_model1(model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def load_model():
    model = ConvNeuralNet().to(device)
    model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))
    model.eval()
    return model

def classification_metrics(model, dataloader, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print("Classification Report:\n")
    print(report)


def predict_with_gui(model):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if not file_path:
        print("No image selected.")
        return

    image = Image.open(file_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output[0], dim=0)
        top_prob, top_class = torch.topk(probabilities, 5)

    plt.imshow(Image.open(file_path))
    plt.axis('off')
    plt.title(f"Prediction: {classes[top_class[0]]} ({top_prob[0].item()*100:.2f}%)")
    plt.show()
    print("\nTop Predictions:")
    for i in range(5):
        print(f"{i+1}. {classes[top_class[i]]:10s} - {top_prob[i].item()*100:.2f}%")

def main():
    if not os.path.exists("cifar10_cnn.pth"):
        print("Starting CIFAR-10 CNN Training")
        print("Loading data...")
        model = train_model()
    else:
        print("Model file found. Loading...")
        model = load_model()
    
    classification_metrics(model, test_loader, classes)
    predict_with_gui(model)
    
if __name__ == "__main__":
    main()
