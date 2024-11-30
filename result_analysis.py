import torch
import matplotlib.pyplot as plt
import numpy as np

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
    