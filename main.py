from pathlib import Path
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
    # train accuracy:       99.87%
    # validation accuracy:  75.24%
    # test accuracy:        74.84%
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
    # testing accuracy: 58.65%
    # validation accuracy: 59.19%
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
    # training accuracy: %
    # testing accuracy: 64.26%
    # validation accuracy: 64.26%
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
    # training accuracy: %
    # testing accuracy: %
    # validation accuracy: %
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
    # print("Running K Means Cluster Classification")
    # km = KMeansClassifier(data_dir, 50000, 1)
    # km.fit()
    # km.evaluate()


    # # autoencoder
    # classifier = UnsupervisedClassification(dir=data_dir, batch_size=32) #, model_path='./autoencoder_model.pth')
    # classifier.train_autoencoder(epochs=10)
    # classifier.evaluate()


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

if __name__ == "__main__":
    main()


