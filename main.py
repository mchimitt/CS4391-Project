from Supervised.AlexNet import AlexNet
from Supervised.SqueezeNet import SqueezeNet
from pathlib import Path

from Pipelines.get_data import get_data
from Unsupervised.Kmeans import KMeansClassifier
from Unsupervised.Autoencoder import UnsupervisedClassification

def main():
    
    # train_loader, val_loader, test_loader, num_classes = get_data(data_dir, batch_size)

    project_root = Path.cwd()
    data_dir = project_root / "Pipelines" / "Wikiart" / "dataset"
    model_dir = project_root / "Models" / "Supervised"


    an = AlexNet(data_dir, model_dir, 15000, 512, 30, 0.0001, 5e-4)
    # Best Hyperparameters: 
    # train samples: 15000, batch size: 512, epochs: 30, lr: 0.0001, decay: 5e-4
    # validation accuracy: 57.92%
    # test accuracy: 58.32%
    an.train()
    an.test()

    # print("Running SqueezeNet")
    # sn = SqueezeNet(data_dir, model_dir, 15000, 512, 30, 0.0001, 0.5, 5e-4)
    # # Best Hyperparameters: 
    # # train samples: 15000, batch size: 512, epochs: 30, lr: 0.0001, dropout: 0.5, decay: 5e-4
    # # testing accuracy: 58.65%
    # # validation accuracy: 59.19%
    # sn.train()
    # sn.test()

    # print("Running K Means Cluster Classification")
    # km = KMeansClassifier(data_dir, 10000, 1)
    # km.fit()
    # km.evaluate()


    # autoencoder
    # classifier = UnsupervisedClassification(dir=data_dir, batch_size=32) #, model_path='./autoencoder_model.pth')
    # classifier.train_autoencoder(epochs=10)
    # classifier.evaluate()


if __name__ == "__main__":
    main()


