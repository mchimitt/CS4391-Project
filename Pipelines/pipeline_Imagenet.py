import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Ensure the code runs as expected on Windows with multiprocessing
if __name__ == "__main__":

    imagenet_dir = 'imagenet'
    if not os.path.exists(imagenet_dir):
        print(f'{imagenet_dir} does not exist')
    else:
        print(f'{imagenet_dir} exists')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Open the training dataset
    train_dir = os.path.join(imagenet_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    print(f'Train dataset size: {len(train_dataset)}')

    # Extract a batch of images and labels
    images, labels = next(iter(train_loader))

    # Print the shape of the images and labels to confirm batch size and dimensions
    print(f'Batch of images shape: {images.shape}')
    print(f'Batch of labels shape: {labels.shape}')
