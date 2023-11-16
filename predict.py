import torch
import config
from utils import save_image,load_checkpoint
from dataset import MapDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from generator import Generator

def save_some_examples(gen, val_loader, epoch, folder):
    x= val_loader
    x= x.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(x * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()

if __name__ == "__main__":
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
    val_dataset = MapDataset(root_dir=r"C:\Users\dhruv\Untitled Folder\Pix2Pix\ans")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
    save_some_examples(gen, val_loader, 1, folder="ans")