#!/usr/bin/env python3
"""
TCG Card Fine-Tuning Script
Fine-tunes MobileNetV3 on TCG card dataset for better visual search accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

class TCGCardDataset(Dataset):
    """
    Dataset for TCG cards with triplet loss training.
    Each sample is (anchor, positive, negative) for contrastive learning.
    """
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, "**/*.jpg"), recursive=True)
        self.image_paths.extend(glob.glob(os.path.join(image_dir, "**/*.png"), recursive=True))

        # Group by card name (assuming filename contains card identifier)
        self.card_groups = {}
        for path in self.image_paths:
            card_name = os.path.basename(path).split('_')[0]  # Adjust based on naming
            if card_name not in self.card_groups:
                self.card_groups[card_name] = []
            self.card_groups[card_name].append(path)

        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths) * 3  # anchor, positive, negative

    def __getitem__(self, idx):
        # Simple triplet sampling (can be improved)
        card_names = list(self.card_groups.keys())

        # Anchor
        anchor_card = np.random.choice(card_names)
        anchor_path = np.random.choice(self.card_groups[anchor_card])

        # Positive (same card, different image)
        positive_path = np.random.choice(self.card_groups[anchor_card])

        # Negative (different card)
        negative_card = np.random.choice([c for c in card_names if c != anchor_card])
        negative_path = np.random.choice(self.card_groups[negative_card])

        anchor_img = self.transform(Image.open(anchor_path).convert('RGB'))
        positive_img = self.transform(Image.open(positive_path).convert('RGB'))
        negative_img = self.transform(Image.open(negative_path).convert('RGB'))

        return anchor_img, positive_img, negative_img

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def fine_tune_mobilenet(image_dir, epochs=10, batch_size=32, learning_rate=1e-4):
    """
    Fine-tune MobileNetV3 on TCG card dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-trained model
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    # Freeze early layers, fine-tune later layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier and last few conv layers
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    # Replace classifier for embedding output
    model.classifier[3] = nn.Identity()

    model = model.to(device)

    # Dataset and DataLoader
    dataset = TCGCardDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Loss and optimizer
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for anchor, positive, negative in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(".4f")

    # Save fine-tuned model
    torch.save(model.state_dict(), 'mobilenetv3_tcg_finetuned.pth')
    print("✅ Fine-tuned model saved as 'mobilenetv3_tcg_finetuned.pth'")

    return model

if __name__ == "__main__":
    # Usage: python fine_tune.py /path/to/tcg/card/images
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fine_tune.py <image_directory>")
        sys.exit(1)

    image_dir = sys.argv[1]
    fine_tune_mobilenet(image_dir)