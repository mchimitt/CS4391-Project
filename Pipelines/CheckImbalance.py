import matplotlib.pyplot as plt
from Pipelines.get_data import get_data
import os
import pandas as pd
import numpy as np

def CheckImbalance(plot_name):
    # Call get_data function to obtain data splits
    train_loader, val_loader, test_loader, num_classes, train_df, val_df, test_df = get_data("./Pipelines/Wikiart/dataset", batch_size=32, ret_df=True)

    # Function to plot class distribution
    def plot_class_distribution(train_df, val_df, test_df):
        # Combine the label counts from all splits
        train_counts = train_df['label'].value_counts()
        val_counts = val_df['label'].value_counts()
        test_counts = test_df['label'].value_counts()
        
        # train_label_counts = train_df['label'].value_counts()
        # print("Label Distribution:\n", train_label_counts)
        # print("\nClass Imbalance Ratio:")
        # print(train_label_counts / len(train_df))
        
        # Prepare the plot data
        class_labels = list(range(num_classes))  # Assuming labels are numerical starting from 0
        train_class_counts = [train_counts.get(i, 0) for i in class_labels]
        val_class_counts = [val_counts.get(i, 0) for i in class_labels]
        test_class_counts = [test_counts.get(i, 0) for i in class_labels]

        # Plot the bar graph
        x = np.arange(num_classes)
        width = 0.25  # Width of the bars for each dataset

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, train_class_counts, width, label='Train')
        ax.bar(x, val_class_counts, width, label='Validation')
        ax.bar(x + width, test_class_counts, width, label='Test')

        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Class Distribution Across Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join('./ImbalanceGraphs', plot_name))  # Save the plot

    # Call the function to plot the graph
    plot_class_distribution(train_df, val_df, test_df)