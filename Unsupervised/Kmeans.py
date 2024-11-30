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
                # get the data
        self.train_loader, self.val_loader, self.test_loader, self.num_classes = get_data(dir, batch_size, max_train_samples)
        
        # setting up the variables for accuracy and silhouette score
        self.train_acc = 0.0
        self.val_acc = 0.0
        self.test_acc = 0.0
        self.train_sil = 0.0
        self.val_sil = 0.0
        self.test_sil = 0.0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Fixed number of clusters based on the number of classes
        self.n_clusters = self.num_classes

        # set the batch size
        self.batch_size = batch_size

        # setup the feature extractor -> sends the batches through the model to get the features 
        self.feature_extractor = FeatureExtractor()

        # Instantiate the KMeans model
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)



    # extract features
    def extract_features(self, data_loader):
        all_features = []
        all_labels = []

        # loop through each batch
        for images, labels in tqdm(data_loader, desc="Extracting Features"):
            # Move images and labels to GPU
            images = images.to(device)  
            labels = labels.to(device)

            # Extract features on the batch
            features = self.feature_extractor.extract_features(images)

            # Flatten features to 2D array (batch_size, num_features)
            all_features.append(features.view(features.size(0), -1).cpu().numpy())  # Move to CPU for numpy compatibility
            all_labels.extend(labels.cpu().numpy())  # Same for labels

        all_features = np.vstack(all_features)  # Stack all features vertically (samples, features)
        return all_features, np.array(all_labels)

    # classify with kmeans
    def kmeans_classify(self, data_loader):
        # Extract features and true labels
        features, true_labels = self.extract_features(data_loader)

        # Dimensionality reduction using PCA
        pca = PCA(n_components=50)
        pca_features = pca.fit_transform(features)

        # Clustering using KMeans with the number of clusters equal to the number of classes
        # kmeans = KMeans(n_clusters=self.num_classes, random_state=42)
        self.kmeans.fit(pca_features)
        predicted_labels = self.kmeans.labels_

        # Now we map predicted cluster labels to true class labels (using majority voting)
        cluster_to_label = self.map_clusters_to_labels(predicted_labels, true_labels)

        # Calculate accuracy
        accuracy = self.calculate_accuracy(predicted_labels, cluster_to_label, true_labels)
        return predicted_labels, accuracy, pca_features

    # looks for the most common true label, to classify the cluster
    def map_clusters_to_labels(self, predicted_labels, true_labels):
        # For each cluster, find the most common true label
        cluster_to_label = {}
        for cluster in np.unique(predicted_labels):
            # Get the true labels for all data points in this cluster
            cluster_indices = np.where(predicted_labels == cluster)[0]
            cluster_true_labels = true_labels[cluster_indices]
            
            # Find the most frequent true label in this cluster
            most_common_label = np.bincount(cluster_true_labels).argmax()
            cluster_to_label[cluster] = most_common_label

        return cluster_to_label

    # calculates the accuracy!
    def calculate_accuracy(self, predicted_labels, cluster_to_label, true_labels):
        # Map each predicted label to the corresponding true class label
        predicted_true_labels = np.array([cluster_to_label[label] for label in predicted_labels])

        # Calculate the accuracy
        accuracy = accuracy_score(true_labels, predicted_true_labels) * 100
        return accuracy

    def evaluate(self):
        # Get predicted cluster labels and accuracy for train, val, and test sets
        train_labels, train_accuracy, train_features = self.kmeans_classify(self.train_loader)
        val_labels, val_accuracy, val_features = self.kmeans_classify(self.val_loader)
        test_labels, test_accuracy, test_features = self.kmeans_classify(self.test_loader)

        self.train_acc = train_accuracy
        self.val_acc = val_accuracy
        self.test_acc = test_accuracy

        # Print accuracy for each set
        print(f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")

        # Silhouette Score
        train_silhouette = compute_silhouette_score(train_features, train_labels)
        val_silhouette = compute_silhouette_score(val_features, val_labels)
        test_silhouette = compute_silhouette_score(test_features, test_labels)

        self.train_sil = train_silhouette
        self.val_sil = val_silhouette
        self.test_sil = test_silhouette

        print(f"Train Silhouette Score: {train_silhouette}")
        print(f"Validation Silhouette Score: {val_silhouette}")
        print(f"Test Silhouette Score: {test_silhouette}")

# silhouette score
def compute_silhouette_score(features, predicted_labels):
    score = silhouette_score(features, predicted_labels)
    return score
