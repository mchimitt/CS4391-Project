import os

# Set the root directory of your dataset
root_dir = "C:\\Matthew\\ut dallas\\School Years\\4 - Senior Year\\CS 4391 - Computer Vision\\Project\\CS4391-Project\\Pipelines\\Wikiart\\dataset"

# Function to check if all image files exist
def check_files(root_dir):
    count = 0
    hidden_count = 0
    # Loop through all class directories
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        
        if os.path.isdir(class_path):
            print(f"Checking class: {class_dir}")
            
            # Loop through all files in the class directory
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                
                # Check if the file exists
                if not os.path.exists(file_path):
                    print(f"Missing file: {file_path}")
                    count += 1
                if not file_name.startswith('.') and file_name.lower().endswith('.jpg'):
                    hidden_count += 1
                    file_path = os.path.join(class_path, file_name)
                
        else:
            print(f"Skipping non-directory item: {class_dir}")

    print(f"Number of files inaccessible: {count}")
    print(f"Number of hidden files: {hidden_count}")

# Run the check
check_files(root_dir)
