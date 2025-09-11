import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import os
import uuid

# Trying a simple GAN, might do something like DCGAN if I feel like
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
run_id: str = str(uuid.uuid4())
image_size: int = 784    # 28*28 for MNIST
latent_dim: int = 100
hidden_dim: int = 256
batch_size: int = 128
epochs: int = 200
lr: float = 0.0002
b1: float = 0.5
b2: float = 0.999

# To make things faster
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True

# Downloading MNIST dataset and transforming to pytorch tensors
raw_tensor: transforms.Compose = transforms.Compose([transforms.ToTensor()])
train_set_raw: datasets.MNIST = datasets.MNIST(root='./', train=True, download=True, transform=raw_tensor)

images: list = [img.view(-1) for img, _ in train_set_raw]
all_pixels: torch.Tensor = torch.cat(images)
mean: torch.Tensor = all_pixels.mean()
std: torch.Tensor = all_pixels.std()
# print(f'Mean: {mean:.4f}, std: {std:.4f}')

normalizer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.MNIST(root='./', train=True, download=True, transform=normalizer)
val_set = datasets.MNIST(root='./', train=False, download=True, transform=normalizer)

dataloader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


class Generator(nn.Module):
    """
    Base GAN, generator
    """
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim // 2, bias=False),
            nn.BatchNorm1d(hidden_dim // 2, 0.8),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim // 2, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, 0.8),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.BatchNorm1d(hidden_dim * 2, 0.8),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim * 2, image_size, bias=False),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        img.view(img.size(0), 1, 28, 28)
        return img  # Reshape to image format


class Discriminator(nn.Module):
    """
    Base GAN, discriminator
    """
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_shape, hidden_dim * 4),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output a prob [0, 1]
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # Flatten image
        validity = self.model(img_flat)
        return validity


generator = Generator(latent_dim).to(device)
discriminator = Discriminator(image_size).to(device)

adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))


def save_images(idx, n_images=16, grid_size=(4, 4)):
    """
    Saves a batch of images to a file
    :param idx: Number of epochs
    :param n_images: Number of images
    :param grid_size: Columns * Rows
    :return:
    """
    z = torch.randn(n_images, latent_dim).to(device)

    with torch.no_grad():
        generated_images = generator(z).cpu().detach()

    generated_images = 0.5 * generated_images + 0.5

    fig, ax = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[0], grid_size[1]))
    for idx, ax in enumerate(ax.flat):
        ax.imshow(generated_images[idx].reshape(28, 28).squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle(f'Epoch {idx}', fontsize=16)
    plt.tight_layout()

    if not os.path.exists(f'runs/{run_id}'):
        os.makedirs(f'runs/{run_id}')
    plt.savefig(f'./runs/{run_id}/epoch{idx}.png')
    plt.close()


if __name__ == '__main__':
    # Start training
    print(f'Training... (Device: {device})')
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Move real images to the device
            real_imgs = imgs.to(device)
            real_labels = torch.ones(imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(imgs.size(0), 1).to(device)

            optimizer_D.zero_grad()

            real_output = discriminator(real_imgs)
            d_loss_real = adversarial_loss(real_output, real_labels)

            r = torch.randn(imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(r)
            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = adversarial_loss(fake_output, fake_labels)

            # Combined discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # use real labels (all ones) for the loss calculation.
            fake_output = discriminator(fake_imgs)
            g_loss = adversarial_loss(fake_output, real_labels)

            g_loss.backward()
            optimizer_G.step()

        # Print losses and save generated images every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            save_images(epoch + 1)

    print("Done.")
    torch.save(generator.state_dict(), 'generator.pt')
    torch.save(discriminator.state_dict(), 'discriminator.pt')
