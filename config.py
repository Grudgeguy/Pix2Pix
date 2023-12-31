import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = r"D:\DS for GANs\scapes\cityscapes\cityscapes\train"
VAL_DIR = r"D:\DS for GANs\scapes\cityscapes\cityscapes\val"
LEARNING_RATE = 4e-5
BATCH_SIZE = 32
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 50
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DISC = "cityscapes_disc.pth.tar"
CHECKPOINT_GEN = "cityscapes_gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        # A.HorizontalFlip(p=0.5),
        # A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
