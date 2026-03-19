# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Images often contain noise due to poor lighting, camera issues, or transmission errors, which reduces their quality. This project aims to develop a Convolutional Autoencoder that can remove noise from images and produce cleaner versions.

The model will be trained using noisy and clean image pairs so that it learns to reconstruct clear images. The goal is to improve image quality while preserving important details.

## DESIGN STEPS
## STEP 1:
Collect Dataset – Use image dataset and add noise to the images.

## STEP 2:
Preprocess Data – Resize images, normalize pixel values, and split into training and testing sets.

## STEP 3:
Build Model – Create a Convolutional Autoencoder with encoder (Conv + MaxPooling) and decoder (Conv + UpSampling).

## STEP 4:
Train Model – Train the network using noisy images as input and clean images as target.

## STEP 5:
Evaluate Model – Test the model on noisy images to check denoising performance.

## STEP 6:
Display Output – Compare original image, noisy image, and denoised image.

## PROGRAM
### Name: KARTHICK S
### Register Number: 212224230114
```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion =nn.MSELoss()
optimizer =optim.Adam(model.parameters(),lr=1e-3)

# Print model summary
summary(model, input_size=(1, 28, 28))

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print('Name: karthick S')
    print('Register Number: 212224230114')
    for epoch in range(epochs):
        running_loss = 0.0
        for images_, _ in loader:
            images_ = images_.to(device)
            noisy_images = add_noise(images_).to(device)
            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch :{epoch+1}/{epochs},Loss: {running_loss/len(loader):.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print('Name: karthick S')
    print('Register Number: 212224230114')
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)
```

### OUTPUT
<img width="998" height="652" alt="image" src="https://github.com/user-attachments/assets/7393a1d4-5462-43ad-b84e-f9d804f14260" />
<img width="624" height="51" alt="image" src="https://github.com/user-attachments/assets/0ced9cd8-9f8f-4fba-8467-b169506c2376" />
<img width="448" height="154" alt="image" src="https://github.com/user-attachments/assets/295454a7-d743-4ac1-8a72-41a0e8906e57" />
<img width="624" height="51" alt="image" src="https://github.com/user-attachments/assets/0ced9cd8-9f8f-4fba-8467-b169506c2376" />
<img width="1265" height="398" alt="image" src="https://github.com/user-attachments/assets/b1426642-634f-4f39-a553-b298b1306082" />

## RESULT
The Convolutional Autoencoder successfully removed noise from images, producing clearer images that are close to the original clean images. The model effectively learned image features and improved image quality.
