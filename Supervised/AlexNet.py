from Pipelines.get_data import get_data
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
from tqdm import tqdm


class AlexNet():
    def __init__(self, dir='..\\Pipelines\\Wikiart\\dataset', save_dir='Models\\Supervised\\', max_train_samples=None, batch_size=128, num_epochs=10, learn_rate=0.001, decay=1e-4):
        # use GPU if one exists and is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading the dataset")
        # Load the dataset
        self.train_loader, self.val_loader, self.test_loader, num_classes = get_data(dir, batch_size, max_train_samples)
        print("Dataset has been loaded")

        # Initialize the AlexNet Model

        # get the alexnet model
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        # change the last layer to fit the number of classes that we have
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
        
        # set the model to use the GPU
        self.model = self.model.to(self.device)

        # get the number of epochs
        self.num_epochs = num_epochs

        # model save path
        self.model_save_path = save_dir / 'alexnet_model.pth'
        
        # Set up loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate, weight_decay=decay)

    def train(self):
        print("Starting training...")
        total_batches = len(self.train_loader)
        start_time = time.time()

        # loop through epochs
        for epoch in range(self.num_epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            curr_batch = 1

            # loop through batches
            for images, labels in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}"):
                
                # send the batches to the GPU if possible
                images, labels = images.to(self.device), labels.to(self.device)
            
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # update the loss
                running_loss += loss.item() * images.size(0)
                curr_batch+=1

            # Calculate average loss for the epoch
            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
            
            # Validation phase
            val_accuracy = self.evaluate(self.val_loader)
            print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        end_time = time.time()

        # get the time it took to train
        elapsed_time = end_time - start_time
        print(f"Training Time: {elapsed_time}")

        # Save the trained model
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"AlexNet Model saved to {self.model_save_path}")
        
    def evaluate(self, data_loader):
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        
        with torch.no_grad():
            # loop through the batches
            for images, labels in tqdm(data_loader, desc="Running Test"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # get the accuracy
        accuracy = 100 * correct / total
        return accuracy
    
    def test(self):
        # test the model
        print("Evaluating on test data...")
        test_accuracy = self.evaluate(self.test_loader)
        print(f"Test Accuracy: {test_accuracy:.2f}%")