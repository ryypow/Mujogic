"""
Example model architecture.

Replace this with your actual model implementation.
"""

import torch
import torch.nn as nn


class ExampleCNN(nn.Module):
    """
    Example Convolutional Neural Network.

    Args:
        num_classes (int): Number of output classes
        input_channels (int): Number of input channels (default: 3 for RGB)
    """

    def __init__(self, num_classes=10, input_channels=3):
        super(ExampleCNN, self).__init__()

        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Quick test
    model = ExampleCNN(num_classes=10)
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 RGB images
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
