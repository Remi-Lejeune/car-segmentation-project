import glob

from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, files, size=None):

        self.files = files

        if size is not None and size < len(files):
            self.files = self.files[:size]

        if size is not None and size < len(self.files):
            self.files = self.files[:size]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]


def files_name(path='carseg_data/clean_data/*.np[yz]'):
    return glob.glob(path)
