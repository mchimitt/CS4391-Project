# CS4391-Project

<hr/>

## Setup
To download the dataset, run the setup.py script
```python setup.py```

This will 
- Download the dataset from Kagglehub
    - The dataset can be found here: [Wikiart Dataset](https://www.kaggle.com/datasets/steubk/wikiart)
- Format the directories correctly
- Remove unnecessary classes to shrink the dataset
- Check for class imbalance and augment the data eliminate the imbalance
    - In the ImbalanceGraphs folder, barcharts showing imbalance before and after augmentation can be seen.

To install libraries, run the requirements.txt file
```pip install -r requirements.txt```

The Models folder is where the code will save models after training.
It also stores graphs that show training and validation accuracy during training.

To train the models, run main.py
```python main.py```

This will train 6 models, 4 supervised and 2 unsupervised (this will take a long time)
- AlexNet
- SqueezeNet
- VGGNet
- VisionTransformer
- Kmeans Clustering with a pretrained model
- Kmeans Clustering with an Autoencoder

<hr/>

## Summary of our Results

### Supervised Summary

#### AlexNet
**Hyperparameters:**
Max Training Samples: 50000
Batch Size: 512 
Epochs: 15
Learn Rate: 0.0001  
Decay: 5e-10

**Accuracies**
Training Accuracy:      99.83%
Validation Accuracy:    75.28%
Testing Accuracy:       74.14%

<hr/>

#### SqueezeNet
**Hyperparameters:**
Max Training Samples: 50000 
Batch Size: 256
Epochs: 30
Learn Rate: 0.0001
Dropout: 0.5
Decay: 5e-10

**Accuracies**
Training Accuracy:      80.58%
Validation Accuracy:    65.91%
Testing Accuracy:       65.56%

<hr/>

#### VGGNet
**Hyperparameters:**
Max Training Samples: 50000
Batch Size: 32
Epochs: 10
Learn Rate: 1e-05
Decay: 5e-10

**Accuracies**
Training Accuracy:      99.82%
Validation Accuracy:    68.57%
Testing Accuracy:       68.42%

<hr/>

#### VisionTransformer
**Hyperparameters:**
Max Training Samples: 50000
Batch Size: 32 
Epochs: 10
Learn Rate: 1e-05
Decay: 5e-10

**Accuracies**
Training Accuracy:      99.53%
Validation Accuracy:    77.58%
Testing Accuracy:       76.58%

<hr/>

### Unsupervised Summary

#### K Means Clustering
**Hyperparameters:**
Max Training Samples: 50000
Batch Size: 1

**Accuracies**
Training Accuracy:      27.07%
Validation Accuracy:    27.13%
Testing Accuracy:       27.48%

**Silhouette Scores**
Train Silhouette Score: 0.07079434394836426
Validation Silhouette Score: 0.07262030243873596
Test Silhouette Score: 0.07275499403476715

<hr/>

#### K Means Clustering with an Autoencoder
**Hyperparameters:**
Max Training Samples: 50000 
Batch Size: 32

**Accuracies**
Training Accuracy:      21.14%

Validation Accuracy:    21.64%

Testing Accuracy:       20.68%

**Silhouette Scores**
Train Silhouette Score: 0.07064420729875565
Validation Silhouette Score: 0.08092181384563446
Test Silhouette Score: 0.0824938490986824

<hr/>
