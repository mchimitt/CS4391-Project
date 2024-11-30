import kagglehub
import shutil
import os
from Pipelines.remover import remover
from Pipelines.CheckImbalance import CheckImbalance
from Pipelines.augment_data import augment_to_balance_classes

def main():

    # Download latest version
    path = kagglehub.dataset_download("steubk/wikiart")
    print("Path to dataset files:", path)
    
    # move the contents of the of the folder to the dataset folder
    move_folders(src_path='/home/chimitt/.cache/kagglehub/datasets/steubk/wikiart/versions/1', dest_path='./Pipelines/Wikiart/dataset', csv_path='./Pipelines/Wikiart/ExcelSheets')

    # filter out some of the classes (make the dataset smaller)
    print("Filtering Classes...")
    remover('./Pipelines/Wikiart/dataset')

    # save class imbalance before augmentation
    CheckImbalance("imbalance_before_augmentation")
    print("Imbalance Graph before Augmentation saved.")

    # augment data to fix class imbalance
    print("Augmenting Data to fix imbalance")
    augment_to_balance_classes('./Pipelines/Wikiart/dataset', max_count=7000)

    # check for imbalance after augmentation
    CheckImbalance("imbalance_after_augmentation")
    print("Imbalance Graph after Augmentation saved.")


def move_folders(src_path, dest_path, csv_path):
    # Check if the destination directory exists, create it if it doesn't
    os.makedirs(dest_path, exist_ok=True)

    # Loop through files/folders in the source path
    for item in os.listdir(src_path):
        src_item = os.path.join(src_path, item)
        dest_item = os.path.join(dest_path, item)
        
        if os.path.isdir(src_item):  # Only copy directories
            shutil.move(src_item, dest_item)
            print(f"Copied folder: {src_item} to {dest_item}")
        else:
            shutil.move(src_item, csv_path)
            print(f"Copied {src_item} to {csv_path}")

if __name__ == '__main__':
    main()