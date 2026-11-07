
import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

NUM_CLASSES = 7


def build_model(pretrained=True, freeze_backbone=True):
    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained else None
    )

    model = resnet18(weights=weights)

    # Freeze pretrained feature extractor initially.
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # Replace original 1000-class classifier.
    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


if __name__ == "__main__":
    print("=" * 55)
    print("LiverPath AI - ResNet-18 Model Test")
    print("=" * 55)

    device = get_device()
    print("Device:", device)

    model = build_model(
        pretrained=True,
        freeze_backbone=True
    ).to(device)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", f"{total_params:,}")
    print("Trainable parameters:", f"{trainable_params:,}")

    # Test forward pass.
    model.eval()

    sample = torch.randn(
        4, 3, 224, 224,
        device=device
    )

    with torch.inference_mode():
        output = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)

    assert output.shape == (4, NUM_CLASSES)

    probabilities = torch.softmax(
        output, dim=1
    )

    print(
        "Probability sums:",
        probabilities.sum(dim=1).cpu().tolist()
    )

    print("\nResNet-18 model test successful!")
