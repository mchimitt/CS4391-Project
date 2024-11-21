from Pipelines.get_data import get_data
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Feature Extractor Class
class FeatureExtractor():
    def __init__(self):
        # Use a pretrained VGG19 model with batch normalization
        self.model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        
        # Remove the classification layer, keep features
        # we are clustering on the features
        self.model.fc = nn.Identity()
        self.model.eval()
        
        # Move the model to the GPU if possible
        self.model = self.model.to(device)

    def extract_features(self, image):
        with torch.no_grad():
            # Ensure image tensor is on the same device as the model
            image = image.to(device).float()
            features = self.model(image)
        return features

class KMeansClassifier():
    def __init__(self, dir='..\\Pipelines\\Wikiart\\dataset', max_train_samples=None, batch_size=128):
        print("Loading the dataset")
        # Load the dataset
        self.train_loader, self.val_loader, self.test_loader, num_classes = get_data(dir, batch_size, max_train_samples)
        print("Dataset has been loaded")

        # Fixed number of clusters based on the number of classes
        self.n_clusters = num_classes

        # set the batch size
        self.batch_size = batch_size

        # setup the feature extractor -> sends the batches through the model to get the features 
        self.feature_extractor = FeatureExtractor()

        # Instantiate the KMeans model
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)

    def extract(self, dataloader):
        all_features = []
        all_labels = []

        # loop through each batch
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)  # Move images to GPU
            labels = labels.to(device)

            # Extract features on the batch
            features = self.feature_extractor.extract_features(images)

            # Flatten features to 2D array (batch_size, num_features)
            all_features.append(features.view(features.size(0), -1).cpu().numpy())  # Move to CPU for numpy compatibility
            all_labels.extend(labels.cpu().numpy())  # Same for labels

        all_features = np.vstack(all_features)  # Stack all features vertically (samples, features)
        return all_features, np.array(all_labels)
    
    def fit(self):
        # get the features
        self.train_features, self.train_labels = self.extract(self.train_loader)

        # Normalize features using StandardScaler
        scaler = StandardScaler()
        self.train_features = scaler.fit_transform(self.train_features)

        # Reduce dimensionality with PCA
        pca = PCA(n_components=50)  # Keep more components if necessary
        self.train_features = pca.fit_transform(self.train_features)

        # Fit the KMeans model
        print("Fitting KMeans...")
        self.kmeans.fit(self.train_features)

    def predict(self):
        # get the features
        test_features = self.train_features  # Using the same training features for clustering
        
        # Predict cluster labels
        cluster_labels = self.kmeans.predict(test_features)
        return cluster_labels

    def evaluate(self):
        # Get predictions from the model
        cluster_labels = self.predict()

        # Map clusters to true labels using majority voting
        cluster_to_true_label = {}
        # loop through each cluster
        for cluster in np.unique(cluster_labels):
            mask = (cluster_labels == cluster)
            most_common = np.bincount(self.train_labels[mask]).argmax()
            cluster_to_true_label[cluster] = most_common
        
        adjusted_labels = [cluster_to_true_label[cluster] for cluster in cluster_labels]
        accuracy = accuracy_score(self.train_labels, adjusted_labels)
        accuracy *= 100  # Convert to percentage
        print(f"Clustering Accuracy: {accuracy:.2f}%")

        # Evaluate clustering quality using Silhouette Score (higher is better)
        silhouette_avg = silhouette_score(self.train_features, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.4f}")

        return accuracy, silhouette_avg
