import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms
from PIL import Image
import os
from collections import defaultdict
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


