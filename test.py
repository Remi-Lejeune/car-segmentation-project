import numpy as np

from image_dataset import *
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from segmentation_model import SegmentationModel
import matplotlib.pyplot as plt


def plot_comparison(reference_path, predicted_path, image_path):
    predicted_mask = np.load(predicted_path)[0].astype(np.float)
    reference_mask = np.load(reference_path)[3]
    image = plt.imread(image_path)
    for i, title in zip(range(3), ["image", "reference", "predicted"]):
        plt.imshow(image[:, 256:, :])
        if i == 1:
            plt.imshow(reference_mask, vmax=reference_mask.max(), alpha=0.7)
        if i == 2:
            plt.imshow(predicted_mask, vmax=reference_mask.max(), alpha=0.7)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(predicted_path.split(".")[0]+"_"+title+".svg", bbox_inches='tight', pad_inches=0, format='svg')
        plt.show()


best_path = "carseg_data/clean_data/52_a.npy"
worst_path = "carseg_data/clean_data/33_a.npy"

plot_comparison(best_path, predicted_path="best_sample.npy", image_path="carseg_data/carseg_raw_data/train/photo/52_a.jpg")
plot_comparison(worst_path, predicted_path="worst_sample.npy", image_path="carseg_data/carseg_raw_data/train/photo/33_a.jpg")


