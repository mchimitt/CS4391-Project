from pathlib import Path
import os

from Pipelines.get_data import get_data

from Supervised.AlexNet import AlexNet
from Supervised.SqueezeNet import SqueezeNet
from Supervised.VGGNet import VGGNet
from Supervised.VisionTransformer import ViTNet

from Unsupervised.Kmeans import KMeansClassifier
from Unsupervised.Autoencoder import UnsupervisedClassification

def main():
    
    # train_loader, val_loader, test_loader, num_classes = get_data(data_dir, batch_size)

    project_root = Path.cwd()
    data_dir = project_root / "Pipelines" / "Wikiart" / "dataset"
    model_dir = project_root / "Models" / "Supervised"

    ########################################################################################################
    ##                                            SUPERVISED                                              ##                                                    
    ########################################################################################################

    # AlexNet
    # Best Hyperparameters: 
    # train samples: 50000, batch size: 512, epochs: 15, lr: 0.0001, decay: 5e-10
    # train accuracy:       99.83%
    # validation accuracy:  75.28%
    # test accuracy:        74.14%
    anTrainingSamples = 50000
    anBatchSize = 512
    anEpochs = 15
    anLearningRate = 0.0001
    anDecay = 5e-10
    print("Running AlexNet")
    an = AlexNet(data_dir, model_dir, max_train_samples=anTrainingSamples, batch_size=anBatchSize, num_epochs=anEpochs, learn_rate=anLearningRate, decay=anDecay)
    an.train()
    an.test()


    # SqueezeNet
    # Best Hyperparameters: 
    # train samples: 50000, batch size: 256, epochs: 30, lr: 0.0001, dropout: 0.5, decay: 5e-10
    # training accuracy: 80.58%
    # testing accuracy: 65.91%
    # validation accuracy: 65.56%
    print("Running SqueezeNet")
    snTrainingSamples = 50000
    snBatchSize = 256
    snEpochs = 30
    snLearningRate = 0.0001
    snDropout = 0.5
    snDecay = 5e-10
    sn = SqueezeNet(data_dir, model_dir, max_train_samples=snTrainingSamples, batch_size=snBatchSize, num_epochs=snEpochs, learn_rate=snLearningRate, dropout=snDropout, decay=snDecay)
    sn.train()
    sn.test()


    # VGGNet
    # Best Hyperparameters: 
    # train samples: 50000, batch size: 32, epochs: 10, lr: 0.00001, decay: 5e-10
    # training accuracy: 99.82%
    # testing accuracy: 68.82%
    # validation accuracy: 68.42%
    print("Running VGGNet")
    vnTrainingSamples = 50000
    vnBatchSize = 32
    vnEpochs = 10
    vnLearningRate = 0.00001
    vnDecay = 5e-10
    vn = VGGNet(data_dir, model_dir, max_train_samples=vnTrainingSamples, batch_size=vnBatchSize, num_epochs=vnEpochs, learn_rate=vnLearningRate, decay=vnDecay)
    vn.train()
    vn.test()


    # Vision Transformer
    # Best Hyperparameters: 
    # train samples: 50000, batch size: 32, epochs: 10, lr: 0.00001, dropout: 0.2, decay: 5e-10
    # training accuracy: 99.53%
    # validation accuracy: 77.58%
    # testing accuracy: 76.58%
    vitTrainingSamples = 50000
    vitBatchSize = 32
    vitEpochs = 10
    vitLearningRate = 0.00001
    vitDecay = 5e-10
    vit = ViTNet(data_dir, model_dir, max_train_samples=vitTrainingSamples, batch_size=vitBatchSize, num_epochs=vitEpochs, learn_rate=vitLearningRate, decay=vitDecay)
    vit.train()
    vit.test()





    ########################################################################################################
    ##                                            UNSUPERVISED                                            ##                                                    
    ########################################################################################################

    # kmeans
    print("Running K Means Cluster Classification")
    km_batch_size = 1
    km_train_samples = 50000
    km = KMeansClassifier(data_dir, max_train_samples=km_train_samples, batch_size=km_batch_size)
    km.evaluate()


    # autoencoder
    print("Running Kmeans Clustering with an Autoencoder")
    auto_max_samples = 50000
    auto_batch_size = 32
    auto_model_path = './Models/Unsupervised/autoencoder_model.pth'
    
    # USING THE TRAINED MODEL (INSTEAD OF RETRAINING IT)
    classifier = UnsupervisedClassification(dir=data_dir, max_train_samples=auto_max_samples, batch_size=auto_batch_size, model_path=auto_model_path)
    
    # # OR RETRAIN THE MODEL AND THEN CLUSTER
    # classifier = UnsupervisedClassification(dir=data_dir, max_train_samples=auto_max_samples, batch_size=auto_batch_size)
    # classifier.train_autoencoder(epochs=10)

    classifier.evaluate()







    print("\n\n\n================SUPERVISED SUMMARY================\n")
    print("AlexNet")
    print("Hyperparameters:")
    print(f"Max Training Samples: {anTrainingSamples}   --  Batch Size: {anBatchSize}   --  Epochs: {anEpochs}")
    print(f"Learn Rate: {anLearningRate}    --  Decay: {anDecay}")
    print(f"\nTraining Accuracy:      {an.train_acc}%")
    print(f"Validation Accuracy:    {an.val_acc}%")
    print(f"Testing Accuracy:       {an.test_acc}%")

    print("\n\n")

    print("SqueezeNet")
    print("Hyperparameters:")
    print(f"Max Training Samples: {snTrainingSamples}   --  Batch Size: {snBatchSize}   --  Epochs: {snEpochs}")
    print(f"Learn Rate: {snLearningRate}    --  Dropout: {snDropout}    --  Decay: {snDecay}")
    print(f"\nTraining Accuracy:      {sn.train_acc}%")
    print(f"Validation Accuracy:    {sn.val_acc}%")
    print(f"Testing Accuracy:       {sn.test_acc}%")

    print("\n\n")

    print("VGGNet")
    print("Hyperparameters:")
    print(f"Max Training Samples: {vnTrainingSamples}   --  Batch Size: {vnBatchSize}   --  Epochs: {vnEpochs}")
    print(f"Learn Rate: {vnLearningRate}    --  Decay: {vnDecay}")
    print(f"\nTraining Accuracy:      {vn.train_acc}%")
    print(f"Validation Accuracy:    {vn.val_acc}%")
    print(f"Testing Accuracy:       {vn.test_acc}%")

    print("\n\n")

    print("VisionTransformer")
    print("Hyperparameters:")
    print(f"Max Training Samples: {vitTrainingSamples}   --  Batch Size: {vitBatchSize}   --  Epochs: {vitEpochs}")
    print(f"Learn Rate: {vitLearningRate}    --  Decay: {vitDecay}")
    print(f"\nTraining Accuracy:      {vit.train_acc}%")
    print(f"Validation Accuracy:    {vit.val_acc}%")
    print(f"Testing Accuracy:       {vit.test_acc}%")

    print("\n==================================================\n")


    print("\n\n\n===============UNSUPERVISED SUMMARY===============\n")
    print("K Means Clustering")
    print("Hyperparameters:")
    print(f"Max Training Samples: {km_train_samples}   --  Batch Size: {km_batch_size}")
    print("\nACCURACIES")
    print(f"Training Accuracy:      {km.train_acc}%")
    print(f"Validation Accuracy:    {km.val_acc}%")
    print(f"Testing Accuracy:       {km.test_acc}%")
    print("\nSILHOUETTE SCORES")
    print(f"Train Silhouette Score: {km.train_sil}")
    print(f"Validation Silhouette Score: {km.val_sil}")
    print(f"Test Silhouette Score: {km.test_sil}")

    print("\n\n")

    print("K Means Clustering with an Autoencoder")
    print("Hyperparameters:")
    print(f"Max Training Samples: {auto_max_samples}   --  Batch Size: {auto_batch_size}")
    print("\nACCURACIES")
    print(f"Training Accuracy:      {classifier.train_acc}%")
    print(f"Validation Accuracy:    {classifier.val_acc}%")
    print(f"Testing Accuracy:       {classifier.test_acc}%")
    print("\nSILHOUETTE SCORES")
    print(f"Train Silhouette Score: {classifier.train_sil}")
    print(f"Validation Silhouette Score: {classifier.val_sil}")
    print(f"Test Silhouette Score: {classifier.test_sil}")

    print("\n==================================================\n")


if __name__ == "__main__":
    main()


