import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Simple U-Net Model (Shallow for Testing)
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load Dataset (Ensure images are small)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root="data/raw/", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize Model, Optimizer, Loss
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training Loop (Small Number of Steps)
for epoch in range(3):
    for imgs, _ in dataloader:
        imgs = imgs.to(device)

        # Apply Gaussian Noise (Simulating Diffusion Process)
        noise = torch.randn_like(imgs) * 0.1  # Low noise for simplicity
        noisy_imgs = imgs + noise

        # Train U-Net to Remove Noise
        preds = model(noisy_imgs)
        loss = criterion(preds, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Generate a Simple Output
def generate_sample(model):
    print("TEST")
    model.eval()
    with torch.no_grad():
        noise = torch.randn(1, 3, 64, 64).to(device)  # Random noise as input
        output = model(noise).cpu().squeeze(0).numpy()

        # Convert to Image Format
        output = np.transpose(output, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        output = np.clip(output, 0, 1)  # Ensure values are valid for display

        plt.imshow(output)
        plt.axis("off")
        plt.title("Generated Sample")
        plt.show()

generate_sample(model)  # View a sample output
