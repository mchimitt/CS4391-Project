import os
from tqdm import tqdm

def delete_augmented_images(dataset_dir, augmentation_prefix="aug_"):
    """
    Deletes augmented images in the dataset directory based on a naming prefix.
    
    Parameters:
    dataset_dir (str): Path to the dataset directory containing class subdirectories.
    augmentation_prefix (str): Prefix used in augmented image filenames (default is 'aug_').
    """
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_dir):
            continue  # Skip if it's not a directory

        # Iterate through files in the class directory
        for img_name in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
            if img_name.startswith(augmentation_prefix):
                # Delete augmented image
                img_path = os.path.join(class_dir, img_name)
                os.remove(img_path)

    print(f"Augmented images with prefix '{augmentation_prefix}' have been deleted from {dataset_dir}")

# Example usage
dataset_dir = './Wikiart/dataset'  # Replace with the path to your dataset
delete_augmented_images(dataset_dir)