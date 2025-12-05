"""
Custom Dataset implementation.

Replace this with your actual dataset class.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ExampleDataset(Dataset):
    """
    Example PyTorch Dataset for image classification.

    Args:
        root_dir (str): Path to dataset directory
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Optional transform to apply to images
    """

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Build file list
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(split='train', img_size=224):
    """
    Get data transforms for train/val/test splits.

    Args:
        split (str): 'train', 'val', or 'test'
        img_size (int): Target image size

    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    # Quick test
    # dataset = ExampleDataset('data/processed', split='train', transform=get_transforms('train'))
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Classes: {dataset.classes}")
    pass
