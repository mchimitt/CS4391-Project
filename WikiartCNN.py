import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Custom Dataset for loading image data.

        Parameters:
        dataframe (pd.DataFrame): DataFrame with columns ['image_path', 'label'].
        transform (callable, optional): Optional transform to be applied on an image.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]  # Image path
        label = self.dataframe.iloc[idx, 1]  # Label

        # Load the image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data(dir, batch_size, max_train_samples=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset paths and labels into pandas DataFrame
    image_paths = []
    labels = []
    class_names = os.listdir(dir)
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dir, class_name)
        for img_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_idx)

    df = pd.DataFrame({'image_path': image_paths, 'label': labels})

    # Stratified Sampling to limit training samples if max_train_samples is specified
    if max_train_samples:
        # Shuffle the DataFrame before stratified sampling
        df = shuffle(df)
        class_distribution = df['label'].value_counts()
        samples_per_class = max_train_samples // len(class_distribution)
        sampled_df = pd.DataFrame()

        for class_label in class_distribution.index:
            class_df = df[df['label'] == class_label]
            sampled_class_df = class_df.sample(min(samples_per_class, len(class_df)), random_state=42)
            sampled_df = pd.concat([sampled_df, sampled_class_df], axis=0)

        df = sampled_df

    # Split dataset into train, validation, and test sets (80%, 10%, 10%)
    train_df, test_val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df['label'], random_state=42)

    # Create custom datasets for each set
    train_dataset = CustomImageDataset(train_df, transform)
    val_dataset = CustomImageDataset(val_df, transform)
    test_dataset = CustomImageDataset(test_df, transform)

    # Create DataLoaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, len(class_names)


class GenreClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(GenreClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
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

        # Validate the model at the end of each epoch
        val_accuracy = validate_model(model, val_loader, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%')


def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


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
    return 100 * correct / total


def main():
    data_dir = "wiki100/dataset"
    batch_size = 125
    max_train_samples = 11000  # Optional: Set None if not limiting samples
    num_epochs = 40
    learning_rate = 0.001
    model_path = "genre_classifier_cnn.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, num_classes = get_data(data_dir, batch_size, max_train_samples)

    # Initialize model, criterion, and optimizer
    model = GenreClassifierCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

    # Test the model
    print("Evaluating on test dataset...")
    test_accuracy = test_model(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
