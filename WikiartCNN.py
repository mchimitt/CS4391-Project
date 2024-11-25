import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image


class GenreClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(GenreClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_data_loader(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader, dataset.classes


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

        # Test the model at the end of each epoch
        test_accuracy = test_model(model, test_loader, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%')


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy


def predict_image(model, img_path, transform, class_names, device):
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]


def main():
    # Set data directory and parameters
    wikiart = "wiki100"
    data_dir = os.path.join(wikiart, 'dataset')
    test_dir = "Wikiart/dataset"
    test_image_path = 'MINIMALISM.jpg'

    batch_size = 128
    num_epochs = 30
    learning_rate = 0.001
    model_path = "genre_classifier_cnn.pth"

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training and test data
    train_loader, class_names = get_data_loader(data_dir, batch_size=batch_size)
    test_loader, _ = get_data_loader(test_dir, batch_size=batch_size)

    # Initialize model, criterion, and optimizer
    model = GenreClassifierCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load or train model
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded from file.")
    else:
        print("Starting training...")
        train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}.")

    # Test the model on a single image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    predicted_genre = predict_image(model, test_image_path, transform, class_names, device)
    print(f"The genre is: {predicted_genre}")


if __name__ == "__main__":
    main()
