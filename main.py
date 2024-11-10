from Supervised.AlexNet import AlexNet
from Supervised.SqueezeNet import SqueezeNet
from pathlib import Path

def main():
    
    project_root = Path.cwd()
    data_dir = project_root / "Pipelines" / "Wikiart" / "dataset"
    model_dir = project_root / "Models" / "Supervised"
    # an = AlexNet(10, 0.00001, 2048, 15000 data_dir)
    # an.train()

    print("Running SqueezeNet")
    sn = SqueezeNet(data_dir, model_dir, 15000, 512, 10, 0.0001, 0.5, 5e-4)
    # current best: 5, 0.0001, 512, 0.001, 1e-6, 15000   --> 48.67%
    # 5, 0.0001, 512, 0.00001, 1e-8, 15000
    sn.train()
    sn.test()

if __name__ == "__main__":
    main()


