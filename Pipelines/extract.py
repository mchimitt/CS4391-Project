import os
import tarfile


def extract_tar_files(directory):
    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.tar'):
            file_path = os.path.join(directory, filename)  # Full path to the tar file
            print(f'Extracting {filename}...')

            # Create a directory for the extracted files
            extract_path = os.path.join(directory, filename[:-4])  # Remove '.tar' from filename for folder name
            os.makedirs(extract_path, exist_ok=True)  # Create directory if it doesn't exist

            # Open and extract the tar file
            with tarfile.open(file_path, 'r') as tar:
                tar.extractall(path=extract_path)  # Extract to the specified directory
                print(f'Extracted to {extract_path}')


# Specify the directory containing the .tar files
directory = 'ILSVRC2012_img_train_t3'  # Change this to your directory
if not os.path.exists(directory):
        print(f'{directory} is not exist')
else:
        print(f'{directory} is exist')
extract_tar_files(directory)
