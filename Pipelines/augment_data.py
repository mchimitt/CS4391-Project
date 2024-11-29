import os
from collections import Counter
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random

def augment_to_balance_classes(dataset_dir, max_count=None):
    # Define augmentation transformations (without ToTensor)
    augmentation_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomCrop(224, padding=4),
    ])

    # 1. Compute class distributions
    class_counts = Counter()
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
    
    if max_count == None:
        # 2. Find the maximum number of images in any class
        max_count = max(class_counts.values())
        print(max_count)
    
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
        
        elif count > max_count:
            # Remove images if the count exceeds the limit
            num_images_to_remove = count - max_count
            class_dir = os.path.join(dataset_dir, class_name)
            original_images = os.listdir(class_dir)

            # Ensure we do not attempt to remove more images than available
            num_images_to_remove = min(num_images_to_remove, len(original_images))

            print(f"Removing {num_images_to_remove} images from class '{class_name}' to reach the limit...")
            
            # Shuffle the images and remove the excess ones
            random.shuffle(original_images)

            for img_name in tqdm(original_images[:num_images_to_remove], desc=f"Removing {class_name}"):
                img_path = os.path.join(class_dir, img_name)
                
                # Try removing the file and check if it exists
                if os.path.exists(img_path):
                    os.remove(img_path)
                else:
                    print(f"Error: File not found {img_path}")

    print("Augmentation complete. Class distributions are now balanced.")


if __name__ == '__main__':
    dataset_dir = './Wikiart/dataset'  # Original dataset path
    augment_to_balance_classes(dataset_dir)