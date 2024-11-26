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

    AlexNet
    print("Running AlexNet")
    an = AlexNet(data_dir, model_dir, 50000, 512, 15, 0.0001, 5e-10)
    # Best Hyperparameters: 
    # train samples: 50000, batch size: 512, epochs: 15, lr: 0.0001, decay: 5e-10
    # train accuracy:       99.87%
    # validation accuracy:  75.24%
    # test accuracy:        74.84%
    an.train()
    an.test()


    # SqueezeNet
    print("Running SqueezeNet")
    sn = SqueezeNet(data_dir, model_dir, 50000, 256, 30, 0.0001, 0.5, 5e-10)
    # Best Hyperparameters: 
    # train samples: 50000, batch size: 256, epochs: 30, lr: 0.0001, dropout: 0.5, decay: 5e-10
    # testing accuracy: 58.65%
    # validation accuracy: 59.19%
    sn.train()
    sn.test()


    # AlexNet
    # Best Hyperparameters: 
    # train samples: 50000, batch size: 512, epochs: 15, lr: 0.0001, decay: 5e-10
    # train accuracy:       99.87%
    # validation accuracy:  75.24%
    # test accuracy:        74.84%
    print("Running AlexNet")
    an = AlexNet(data_dir, model_dir, 50000, 512, 15, 0.0001, 5e-10)
    an.train()
    an.test()



    # VGGNet
    # Best Hyperparameters: 
    # train samples: 50000, batch size: 32, epochs: 10, lr: 0.00001, dropout: 0.2, decay: 5e-10
    # training accuracy: %
    # testing accuracy: 64.26%
    # validation accuracy: 64.26%
    print("Running VGGNet")
    vn = VGGNet(data_dir, model_dir, 50000, 32, 10, 0.00001, 0.2, 5e-10)
    vn.train()
    vn.test()


    # Vision Transformer
    # Best Hyperparameters: 
    # train samples: 50000, batch size: 32, epochs: 10, lr: 0.00001, dropout: 0.2, decay: 5e-10
    # training accuracy: %
    # testing accuracy: %
    # validation accuracy: %
    vit = ViTNet(data_dir, model_dir, 50000, 32, 10, 0.00001, 0.2, 5e-10)
    vit.train()
    vit.test()


    print("\n\n\n================SUPERVISED SUMMARY================\n")
    print("AlexNet")
    print(f"Training Accuracy:      {an.train_acc}%")
    print(f"Validation Accuracy:    {an.val_acc}%")
    print(f"Testing Accuracy:       {an.test_acc}%")

    print("\n")

    print("SqueezeNet")
    print(f"Training Accuracy:      {sn.train_acc}%")
    print(f"Validation Accuracy:    {sn.val_acc}%")
    print(f"Testing Accuracy:       {sn.test_acc}%")

    print("VGGNet")
    print(f"Training Accuracy:      {vn.train_acc}%")
    print(f"Validation Accuracy:    {vn.val_acc}%")
    print(f"Testing Accuracy:       {vn.test_acc}%")

    print("VisionTransformer")
    print(f"Training Accuracy:      {vit.train_acc}%")
    print(f"Validation Accuracy:    {vit.val_acc}%")
    print(f"Testing Accuracy:       {vit.test_acc}%")

    print("\n==================================================\n")


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


if __name__ == "__main__":
    main()


