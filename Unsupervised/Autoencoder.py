import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Pipelines.get_data import get_data
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from tqdm import tqdm



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder Model
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output size: (112, 112)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output size: (56, 56)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output size: (28, 28)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output size: (14, 14)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),  # Latent space (256 features)
            nn.ReLU()
        )

        # Decoder Model
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * 14 * 14),
            nn.ReLU(),
            nn.Unflatten(1, (128, 14, 14)),  
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output should be in range [0, 1]
        )

    def forward(self, x):
        # send through the encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def save_model(self, filepath):
        # Save the model to a file
        # allows us to run the clustering algorithm on the model that is already trained
        # don't train it more than once because that takes a while
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        # Load the model from a file
        self.load_state_dict(torch.load(filepath))
        print(f"Model loaded from {filepath}")

    def compute_accuracy(self, outputs, targets):
        # Compute pixel-level accuracy between outputs and targets 
        outputs = outputs > 0.5  # Apply threshold (since output is between 0 and 1)
        targets = targets > 0.5  # Apply threshold to the original image
        correct = (outputs == targets).sum().item()
        total_pixels = torch.numel(outputs)
        accuracy = correct / total_pixels
        return accuracy


# silhouette score
def compute_silhouette_score(features, predicted_labels):
    score = silhouette_score(features, predicted_labels)
    return score

# cluster priority
def cluster_purity(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    return purity

# ari
def compute_ari(true_labels, predicted_labels):
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return ari

class UnsupervisedClassification:
    def __init__(self, dir, batch_size, max_train_samples=None, model_path=None):
        # get the data
        self.train_loader, self.val_loader, self.test_loader, self.num_classes = get_data(dir, batch_size, max_train_samples)

        # setup GPU usage if possible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to GPU or CPU
        self.autoencoder = Autoencoder().to(self.device)  
        
        # Load pre-trained model if path is provided
        if model_path:
            self.autoencoder.load_model(model_path)  
        else:
            self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
            self.loss_fn = nn.MSELoss()
            
        # stats for plotting graph
        self.stats = {
            't': [],
            'loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # saved plot name
        self.plot_file = "autoencoder.pdf"

    # train the autoencoder
    def train_autoencoder(self, epochs=5):
        # set to training mode
        self.autoencoder.train()
        
        # epoch loop
        for epoch in range(epochs):
            running_loss = 0.0
            running_accuracy = 0.0
            i = 0
            
            # loop through the batches
            for data, _ in tqdm(self.train_loader, f"Training Encoder: Epoch {epoch+1}"):
                data = data.to(self.device)  # Move data to GPU or CPU
                self.optimizer.zero_grad()
                output = self.autoencoder(data)
                loss = self.loss_fn(output, data)
                
                # Add loss to plot
                self.stats['t'].append(i / len(self.train_loader) + epoch)
                self.stats['loss'].append(loss.item())
                
                loss.backward()
                self.optimizer.step()

                # Compute accuracy for the current batch
                accuracy = self.autoencoder.compute_accuracy(output, data)
                running_loss += loss.item()
                running_accuracy += accuracy
                i += 1
                
            # get the avg loss and accuracy
            avg_loss = running_loss / len(self.train_loader)
            avg_accuracy = running_accuracy / len(self.train_loader)
            
            # Training accuracy and plot
            _, train_accuracy, _ = self.unsupervised_classification(self.train_loader)
            print(f"Training Accuracy: {train_accuracy:.2f}%")
            self.stats['train_acc'].append(train_accuracy)
            
            # Validation accuracy and plot
            _, val_accuracy, _ = self.unsupervised_classification(self.val_loader)
            print(f"Validation Accuracy: {val_accuracy:.2f}%")
            self.stats['val_acc'].append(val_accuracy)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy*100:.2f}%")
        
        # Plot stats
        print(f'Saving plot to {self.plot_file}')
        self.plot_stats(self.stats, self.plot_file)
        
        # Save the trained model after training
        self.autoencoder.save_model("autoencoder_model.pth")

    # extract features using the autoencoder
    def extract_features(self, data_loader):
        self.autoencoder.eval()
        features = []
        true_labels = []
        with torch.no_grad():
            # loop through the batches
            for data, labels in tqdm(data_loader, "Extracting Features"):
                data = data.to(self.device)
                feature = self.autoencoder.encoder(data)
                features.append(feature.cpu().numpy())
                true_labels.append(labels.numpy())  # Save the true labels for accuracy calculation
        return np.concatenate(features, axis=0), np.concatenate(true_labels, axis=0)

    # classify with kmeans
    def unsupervised_classification(self, data_loader):
        # Extract features and true labels
        features, true_labels = self.extract_features(data_loader)

        # Dimensionality reduction using PCA
        pca = PCA(n_components=50)
        pca_features = pca.fit_transform(features)

        # Clustering using KMeans with the number of clusters equal to the number of classes
        kmeans = KMeans(n_clusters=self.num_classes, random_state=42)
        kmeans.fit(pca_features)
        predicted_labels = kmeans.labels_

        # Now we map predicted cluster labels to true class labels (using majority voting)
        cluster_to_label = self.map_clusters_to_labels(predicted_labels, true_labels)

        # Calculate accuracy
        accuracy = self.calculate_accuracy(predicted_labels, cluster_to_label, true_labels)
        return predicted_labels, accuracy, pca_features

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

    def calculate_accuracy(self, predicted_labels, cluster_to_label, true_labels):
        # Map each predicted label to the corresponding true class label
        predicted_true_labels = np.array([cluster_to_label[label] for label in predicted_labels])

        # Calculate the accuracy
        accuracy = accuracy_score(true_labels, predicted_true_labels) * 100
        return accuracy

    def evaluate(self):
        # Get predicted cluster labels and accuracy for train, val, and test sets
        train_labels, train_accuracy, train_features = self.unsupervised_classification(self.train_loader)
        val_labels, val_accuracy, val_features = self.unsupervised_classification(self.val_loader)
        test_labels, test_accuracy, test_features = self.unsupervised_classification(self.test_loader)

        # Print accuracy for each set
        print(f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")

        # # Evaluate Purity and ARI (as before)
        # true_train_labels = [label for _, label in self.train_loader.dataset]
        # true_val_labels = [label for _, label in self.val_loader.dataset]
        # true_test_labels = [label for _, label in self.test_loader.dataset]

        # train_purity = cluster_purity(true_train_labels, train_labels)
        # val_purity = cluster_purity(true_val_labels, val_labels)
        # test_purity = cluster_purity(true_test_labels, test_labels)

        # print(f"Train Purity: {train_purity}")
        # print(f"Validation Purity: {val_purity}")
        # print(f"Test Purity: {test_purity}")

        # # Adjusted Rand Index
        # train_ari = compute_ari(true_train_labels, train_labels)
        # val_ari = compute_ari(true_val_labels, val_labels)
        # test_ari = compute_ari(true_test_labels, test_labels)

        # print(f"Train ARI: {train_ari}")
        # print(f"Validation ARI: {val_ari}")
        # print(f"Test ARI: {test_ari}")

        # # Silhouette Score
        # train_silhouette = compute_silhouette_score(train_features, train_labels)
        # val_silhouette = compute_silhouette_score(val_features, val_labels)
        # test_silhouette = compute_silhouette_score(test_features, test_labels)

        # print(f"Train Silhouette Score: {train_silhouette}")
        # print(f"Validation Silhouette Score: {val_silhouette}")
        # print(f"Test Silhouette Score: {test_silhouette}")
        
    # Plot the loss and accuracy graphs to the target plot file as a PDF    
    def plot_stats(self, stats, filename):
        plt.subplot(1, 2, 1)
        plt.plot(stats['t'], stats['loss'], 'o', alpha=0.5, ms=4)
        plt.title('Loss')
        plt.xlabel('Epoch')
        loss_xlim = plt.xlim()

        plt.subplot(1, 2, 2)
        epoch = np.arange(1, 1 + len(stats['train_acc']))
        plt.plot(epoch, stats['train_acc'], '-o', label='train')
        plt.plot(epoch, stats['val_acc'], '-o', label='val')
        plt.xlim(loss_xlim)
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.gcf().set_size_inches(12, 4)
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()