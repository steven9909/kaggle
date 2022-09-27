import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from rich.progress import track
from model import PCAModel


def main():

    # data_dir = Path("data/numpynet_test")

    # tensor_l = []

    # for path in track(data_dir.glob("*.npy"), total=100000):
    #     tensor_l.append(np.load(path))

    # pca = PCA(64)
    # pca.fit(tensor_l)

    # with open("pca.pkl", "wb") as f:
    #     pickle.dump(pca, f)

    from PIL import Image
    from torchvision.transforms import functional as TF

    model = PCAModel(Path("pca.pkl")).eval()
    image = Image.open("C:/Users/JBenn/Downloads/NebraskaImage_small.jpg")
    print(model(TF.pil_to_tensor(image).unsqueeze(0)))


if __name__ == "__main__":
    main()
