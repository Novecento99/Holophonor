import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import requests
import tarfile
import time
import os
import torch.nn as nn


# Adjust the generator model
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.dimensions = 100
        self.pixels = 256
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


# Adjust the discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


if __name__ == "__main__":

    # Initialize the models
    device = "cuda"
    generatorNet = Generator().to(device=device)
    discriminatorNet = Discriminator().to(device=device)

    # Hyperparameters
    batch_size = 128
    image_size = generatorNet.pixels
    num_tot_epochs = 1000
    learning_rate = 0.0003
    beta1 = 0.5

    dataset_path = "fiori_images"
    model_name = "flowers_test"
    # Save the models
    model_name = (
        model_name
        + "_"
        + str(image_size)
        + "p_"
        + str(generatorNet.dimensions)
        + "d_"
        + str(batch_size)
        + "e_"
        + str(num_tot_epochs)
    )

    # Update transformations
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    model_directory_name = f"custom_models\{model_name}"
    epoches_image_dir_name = model_directory_name + r"\epoches_images"
    try:
        os.mkdir(model_directory_name)
        print(f"Directory {model_directory_name} created successfully.")
        os.mkdir(epoches_image_dir_name)
        print(f"Directory {epoches_image_dir_name} created successfully.")
    except FileExistsError:
        print(f"Directory already exists.")
        exit()

    # Load the dataset
    try:
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    except:
        print("download images")
        # Download and extract the Oxford 102 Flower Dataset
        url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
        response = requests.get(url)
        with open("102flowers.tgz", "wb") as f:
            f.write(response.content)

        with tarfile.open("102flowers.tgz", "r:gz") as tar_ref:
            tar_ref.extractall(dataset_path)
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(
        discriminatorNet.parameters(), lr=learning_rate, betas=(beta1, 0.999)
    )
    optimizerG = optim.Adam(
        generatorNet.parameters(), lr=learning_rate, betas=(beta1, 0.999)
    )

    # Fixed noise for generating images
    fixed_noise = torch.randn(9, generatorNet.dimensions, 1, 1).to(device=device)

    # Training loop
    epoch_start_time = time.time()
    print("training started")
    for i_epoch in range(num_tot_epochs):
        for i, data in enumerate(dataloader):
            # Update Discriminator
            discriminatorNet.zero_grad()
            real_images = data[0].to(device=device)

            batch_size = real_images.size(0)
            labels = torch.full((batch_size,), 1.0, dtype=torch.float).to(device=device)
            output = discriminatorNet(real_images).view(-1)
            lossD_real = criterion(output, labels)
            lossD_real.backward()

            noise = torch.randn(batch_size, generatorNet.dimensions, 1, 1).to(
                device=device
            )
            fake_images = generatorNet(noise)
            labels.fill_(0.0)
            output = discriminatorNet(fake_images.detach()).view(-1)
            lossD_fake = criterion(output, labels)
            lossD_fake.backward()

            optimizerD.step()

            # Update Generator
            generatorNet.zero_grad()
            labels.fill_(1.0)
            output = discriminatorNet(fake_images).view(-1)
            lossG = criterion(output, labels)
            lossG.backward()

            optimizerG.step()

        training_time = time.time() - epoch_start_time
        print(
            f"Epoch [{i_epoch+1}/{num_tot_epochs}] Loss D: {lossD_real + lossD_fake}, Loss G: {lossG}"
        )
        print(
            f"Training time: {training_time:.2f} seconds. Estimated remaining minutes: {int((training_time/(i_epoch+1))*(num_tot_epochs-i_epoch+1)/60)}"
        )

        print("saving ", model_name)
        # Create the directory

        torch.save(
            generatorNet.state_dict(),
            f"custom_models\{model_name}\generator_" + model_name + ".pth",
        )
        torch.save(
            discriminatorNet.state_dict(),
            f"custom_models\{model_name}\discriminator_" + model_name + ".pth",
        )
        if i_epoch % 2 == 0:
            torch.save(
                generatorNet.state_dict(),
                f"custom_models\{model_name}\generator_"
                + str(i_epoch)
                + model_name
                + ".pth",
            )
            torch.save(
                discriminatorNet.state_dict(),
                f"custom_models\{model_name}\discriminator_"
                + str(i_epoch)
                + model_name
                + ".pth",
            )

        # Generate and save an image
        with torch.no_grad():
            fixed_fake_images = generatorNet(fixed_noise).cpu().detach()
            plt.figure(figsize=(10, 10))
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow((fixed_fake_images[i].permute(1, 2, 0) + 1) / 2)
                plt.axis("off")
            plt.savefig(f"{epoches_image_dir_name}\epoch_{i_epoch+1}.png")
            plt.close()
