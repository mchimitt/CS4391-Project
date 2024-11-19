import os
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import io
from tqdm import tqdm

class new_preprocess_images:
    def __init__(self, dataset_path, target_size=(224, 224), batch_size=1000, train_ratio=0.8, jpeg_quality=95):
        # Convert dataset path to absolute path for reliable directory creation
        self.dataset_path = Path(dataset_path).resolve()
        self.target_size = target_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.jpeg_quality = jpeg_quality
        self.label_encoder = LabelEncoder()

        # Define absolute paths for output directories
        self.train_path = self.dataset_path / "train"
        self.test_path = self.dataset_path / "test"
        self.batch_path = self.dataset_path / "batches"
        
        print(f"Dataset path (absolute): {self.dataset_path}")
        
        # Try creating directories and catch any issues
        try:
            os.makedirs(self.train_path, exist_ok=True)
            os.makedirs(self.test_path, exist_ok=True)
            os.makedirs(self.batch_path, exist_ok=True)
            print("Directories created successfully.")
        except Exception as e:
            print(f"Failed to create directories: {e}")

    def load_and_resize_image(self, image_path):
        """
        Load and resize an image, convert to RGB, and compress to JPEG with uint8.

        Args:
            image_path (str): Path to the image
        
        Returns:
            bytes: JPEG-compressed image in uint8 format
        """
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img = img.resize(self.target_size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
                
                # Convert to numpy array in uint8 format
                img_array = np.array(img, dtype=np.uint8)
                
                # Convert numpy array back to JPEG bytes
                with io.BytesIO() as output:
                    img = Image.fromarray(img_array)
                    img.save(output, format="JPEG", quality=self.jpeg_quality)
                    return output.getvalue()
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None


    def load_data_from_folders(self):
        """
        Load image paths and labels based on the folder structure.

        Returns:
            tuple: (file_paths, labels, label_mapping)
        """
        file_paths = []
        labels = []
        
        for style_folder in self.dataset_path.iterdir():
            if style_folder.is_dir():
                style_name = style_folder.name
                for image_file in style_folder.glob("*.jpg"):
                    file_paths.append(str(image_file))
                    labels.append(style_name)
                    
        # Encode labels based on folder names
        labels = self.label_encoder.fit_transform(labels)
        label_mapping = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
        
        return file_paths, labels, label_mapping

    def batch_process_images(self):
        file_paths, labels, label_mapping = self.load_data_from_folders()
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            file_paths, labels, test_size=1-self.train_ratio, random_state=42
        )

        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.batch_path, exist_ok=True)

        self._process_and_save_batches(train_paths, train_labels, self.train_path, "train")
        self._process_and_save_batches(test_paths, test_labels, self.test_path, "test")

        print("Batch processing completed.")
        print(f"Label mapping: {label_mapping}")

    def _process_and_save_batches(self, file_paths, labels, output_path, split_name):
        num_batches = len(file_paths) // self.batch_size + int(len(file_paths) % self.batch_size != 0)
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(file_paths))
            batch_file_paths = file_paths[batch_start:batch_end]
            batch_labels = labels[batch_start:batch_end]
            
            # Process images as JPEG with uint8
            images = [self.load_and_resize_image(path) for path in batch_file_paths]
            images = [img for img in images if img is not None]
            
            # Save batch files
            np.save(self.batch_path / f"{split_name}_images_batch_{batch_idx}.npy", np.array(images))
            np.save(self.batch_path / f"{split_name}_labels_batch_{batch_idx}.npy", batch_labels)
            
            print(f"Processed and saved batch {batch_idx + 1}/{num_batches} for {split_name} set")
