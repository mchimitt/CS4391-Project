from Supervised.AlexNet import AlexNet
from Supervised.SqueezeNet import SqueezeNet
from pathlib import Path

from Pipelines.get_data import get_data
from Unsupervised.Kmeans import KMeansClassifier

def main():
    
    # train_loader, val_loader, test_loader, num_classes = get_data(data_dir, batch_size)

    project_root = Path.cwd()
    data_dir = project_root / "Pipelines" / "Wikiart" / "dataset"
    model_dir = project_root / "Models" / "Supervised"
    an = AlexNet(data_dir, model_dir, 15000, 512, 10, 0.00001, 5e-4)
    an.train()
    an.test()

    print("Running SqueezeNet")
    # sn = SqueezeNet(data_dir, model_dir, 15000, 512, 10, 0.0001, 0.5, 5e-4)
    # current best: 5, 0.0001, 512, 0.001, 1e-6, 15000   --> 48.67%
    # 5, 0.0001, 512, 0.00001, 1e-8, 15000
    # sn.train()
    # sn.test()

    # print("Running K Means Cluster Classification")
    # km = KMeansClassifier(data_dir, 10000, 16)
    # km.fit()
    # km.evaluate()

if __name__ == "__main__":
    main()


