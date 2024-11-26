import matplotlib.pyplot as plt
from Pipelines.get_data import get_data
import os
import pandas as pd
import numpy as np

# # Load data using get_data
# dir = "./Pipelines/Wikiart/dataset"
# batch_size = 32
# _, _, _, _ = get_data(dir, batch_size, max_train_samples=15000)

# # Calculate and display label distribution
# def check_class_imbalance(df):
#     label_counts = df['label'].value_counts()
#     print("Label Distribution:\n", label_counts)
#     print("\nClass Imbalance Ratio:")
#     print(label_counts / len(df))

#     # Plot the distribution
#     plt.figure(figsize=(10, 5))
#     label_counts.plot(kind='bar')
#     plt.title("Class Distribution")
#     plt.xlabel("Class Labels")
#     plt.ylabel("Frequency")
#     plt.savefig("class_distribution.png")  # Save the plot

# # Assuming df is your dataframe with all data
# image_paths = []
# labels = []
# class_names = os.listdir(dir)
# for class_idx, class_name in enumerate(class_names):
#     class_dir = os.path.join(dir, class_name)
#     for img_name in os.listdir(class_dir):
#         image_paths.append(os.path.join(class_dir, img_name))
#         labels.append(class_idx)

# df = pd.DataFrame({'image_path': image_paths, 'label': labels})
# check_class_imbalance(df)


# import matplotlib.pyplot as plt

# Call get_data function to obtain data splits
train_loader, val_loader, test_loader, num_classes, train_df, val_df, test_df = get_data("./Pipelines/Wikiart/dataset", batch_size=32, ret_df=True, max_train_samples=15000)

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
    plt.savefig("class_distribution.png")  # Save the plot

# Call the function to plot the graph
plot_class_distribution(train_df, val_df, test_df)