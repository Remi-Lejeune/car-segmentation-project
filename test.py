from image_dataset import *
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from segmentation_model import SegmentationModel


test_files = get_test_files()
model = SegmentationModel.load_from_checkpoint(checkpoint_path="lightning_logs/version_2/checkpoints/epoch=999-step=70000.ckpt")
model.eval()

test_dataset = ImageDataset(test_files)
trainer = Trainer(
    accelerator="gpu",
    devices=1,
)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
trainer.test(model, test_dataloader)


"""
files_2 = get_files()
files_a = get_test_files()
files_clean = get_clean_files()
print(files_2)
print(files_a)
print(files_clean)

print(len(files_2))
print(len(files_a))
print(len(files_clean))
"""