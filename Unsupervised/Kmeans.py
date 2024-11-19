from Pipelines.get_data import get_data
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image

# Extract the features of the images and cluster based on these features
class FeatureExtractor():
    def __init__(self):
        # use a pretrained model (resnet in this case)
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # remove the last layer which classifies, we just want the features
        self.model.fc = nn.Identity()
        self.model.eval()

    def extract_features(self, image):
        with torch.no_grad():
            features = self.model(image)
        return features

class KMeansClassifier():
    def __init__(self, dir='..\\Pipelines\\Wikiart\\dataset',  max_train_samples=None, batch_size=128):
        # maybe add this to the parameters: save_dir='Models\\Supervised\\',
        
        print("Loading the dataset")
        # Load the dataset
        self.train_loader, self.val_loader, self.test_loader, num_classes = get_data(dir, batch_size, max_train_samples)
        print("Dataset has been loaded")

        # the number of clusters is the number of classes that we have
        self.n_clusters = num_classes
        self.batch_size = batch_size
        # create the feature extractor
        self.feature_extractor = FeatureExtractor()
        # instaniate the cluster model
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)

    def extract(self, dataloader):
        # storing features and labels
        all_features = []
        all_labels = []

        # extracting each feature from the dataloader
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            # extract the features
            features = self.feature_extractor.extract_features(images)
            # add the features to the feature list
            all_features.append(features.numpy())
            # add the label to the labels list
            all_labels.extend(labels.numpy())

        all_features = np.vstack(all_features)
        # return the list of features and labels
        return all_features, np.array(all_labels)
    
    def fit(self):
        
        # get the features
        self.train_features, self.train_labels = self.extract(self.train_loader)
        
        # use PCA to reduce dimensionality
        # pca = PCA(n_components=50)
        # features_pca = pca.fit_transform(self.train_features)

        # Using our kmeans model to fit the features
        print("Fitting KMeans...")
        # self.kmeans.fit(features_pca)
        self.kmeans.fit(self.train_features)

    def predict(self):
        
        # extract the features 
        # features, labels = self.extract(self.test_loader)
        
        # Apply PCA before predicting (same as used during fitting)
        pca = PCA(n_components=50)
        # features_pca = pca.fit_transform(self.train_features)
        
        # do prediction
        # cluster_labels = self.kmeans.predict(features_pca)
        cluster_labels = self.kmeans.predict(self.train_features)
        return cluster_labels, self.train_labels
    
    def evaluate(self):
        cluster_labels, true_labels = self.predict()

        # Adjust cluster labels to match true labels (using majority voting)
        cluster_to_true_label = {}
        for cluster in np.unique(cluster_labels):
            mask = (cluster_labels == cluster)
            most_common = np.bincount(true_labels[mask]).argmax()
            cluster_to_true_label[cluster] = most_common

        adjusted_labels = [cluster_to_true_label[cluster] for cluster in cluster_labels]
        accuracy = accuracy_score(true_labels, adjusted_labels)

        print(f"Clustering Accuracy: {accuracy:.4f}")
        return accuracy