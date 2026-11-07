
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABELS = [
    "FIB", "INF", "NEC",
    "NOR", "REA", "STE", "TUM"
]

LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}

# ImageNet normalisation for pretrained ResNet.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class LiverPathDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        unknown = (
            set(self.data["label"]) - set(LABELS)
        )
        if unknown:
            raise ValueError(
                f"Unknown tissue labels: {unknown}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = Path(row["filepath"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        label = LABEL_TO_INDEX[row["label"]]

        return image, torch.tensor(
            label, dtype=torch.long
        )


def create_dataloaders(batch_size=32, num_workers=0):
    split_dir = Path("data/processed/splits")

    train_dataset = LiverPathDataset(
        split_dir / "train.csv",
        transform=train_transform
    )

    val_dataset = LiverPathDataset(
        split_dir / "val.csv",
        transform=eval_transform
    )

    test_dataset = LiverPathDataset(
        split_dir / "test.csv",
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("=" * 55)
    print("LiverPath AI - PyTorch Dataset Test")
    print("=" * 55)

    train_loader, val_loader, test_loader = (
        create_dataloaders(batch_size=16)
    )

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("\nFirst batch:")
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Labels:", labels.tolist())

    assert images.shape == (16, 3, 224, 224)
    assert labels.shape == (16,)

    print("\nPyTorch DataLoader test successful!")
