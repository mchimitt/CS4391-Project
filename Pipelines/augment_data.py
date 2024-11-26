import os
from collections import Counter
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random

def augment_to_balance_classes(dataset_dir, augmentation_transforms):
    """
    Augment images in the dataset to balance the class distributions.
    Only augment the minority classes so that each class has the same number of images as the majority class.

    Parameters:
    dataset_dir (str): Directory containing the dataset with class subdirectories.
    augmentation_transforms (transforms.Compose): Transformations for image augmentation.
    """
    # 1. Compute class distributions
    class_counts = Counter()
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
    
    # 2. Find the maximum number of images in any class
    max_count = max(class_counts.values())
    
    print("Class counts before augmentation:")
    print(class_counts)

    # 3. Augment only the minority classes
    for class_name, count in class_counts.items():
        if count < max_count:
            # Number of images needed to balance the class
            num_images_to_augment = max_count - count
            class_dir = os.path.join(dataset_dir, class_name)

            # Get the list of original images in the class
            original_images = os.listdir(class_dir)

            print(f"Augmenting class '{class_name}' with {num_images_to_augment} new images...")
            for _ in tqdm(range(num_images_to_augment), desc=f"Augmenting {class_name}"):
                # Select a random image to augment
                img_name = random.choice(original_images)
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path).convert('RGB')

                # Apply augmentation transformations
                augmented_img = augmentation_transforms(img)

                # Save augmented image with a unique name
                augmented_img_name = f"aug_{random.randint(1000, 9999)}_{img_name}"
                augmented_img.save(os.path.join(class_dir, augmented_img_name))

    print("Augmentation complete. Class distributions are now balanced.")

# Define augmentation transformations (without ToTensor)
augmentation_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomCrop(224, padding=4),
])

dataset_dir = './Wikiart/dataset'  # Original dataset path

augment_to_balance_classes(dataset_dir, augmentation_transforms)