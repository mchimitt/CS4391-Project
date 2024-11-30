import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_class_accuracies(model, test_loader, class_names):
    """
    Generates a bar graph of accuracies by class for a given model and test dataset.
    
    Args:
        model (torch.nn.Module): The trained classification model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        class_names (list): List of class names corresponding to the dataset labels.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize tracking for class-wise metrics
    num_classes = len(class_names)
    correct_predictions = np.zeros(num_classes, dtype=int)
    total_samples = np.zeros(num_classes, dtype=int)
    
    with torch.no_grad():
        for images, labels in test_loader:
            # Forward pass through the model
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            # Update tracking for each class
            for label, prediction in zip(labels, predictions):
                total_samples[label.item()] += 1
                if label.item() == prediction.item():
                    correct_predictions[label.item()] += 1
    
    # Calculate accuracies
    accuracies = (correct_predictions / total_samples) * 100  # Convert to percentages
    
    # Plot bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, accuracies, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-wise Accuracy')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix(model, test_loader, class_names):
    """
    Displays a color-coded confusion matrix for the given model and test dataset.
    
    Args:
        model (torch.nn.Module): The trained classification model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        class_names (list): List of class names corresponding to the dataset labels.
    """
    # Set the model to evaluation mode
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            # Forward pass
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            # Collect predictions and true labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, colorbar=True)
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    