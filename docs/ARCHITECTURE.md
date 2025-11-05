# Model Architecture Documentation

## Overview
Detailed description of the model architecture(s) used in this project.

## Model 1: Example CNN

### Architecture Diagram
```
Input (3, 224, 224)
    ↓
Conv2d(3 → 32, k=3)
    ↓
BatchNorm2d(32)
    ↓
ReLU
    ↓
MaxPool2d(k=2, s=2)
    ↓
Conv2d(32 → 64, k=3)
    ↓
BatchNorm2d(64)
    ↓
ReLU
    ↓
MaxPool2d(k=2, s=2)
    ↓
AdaptiveAvgPool2d(1, 1)
    ↓
Flatten
    ↓
Linear(64 → 128)
    ↓
ReLU
    ↓
Dropout(0.5)
    ↓
Linear(128 → num_classes)
    ↓
Output (num_classes)
```

### Architecture Details

**Input**: RGB images of size 224×224

**Feature Extraction**:
- Conv Block 1: 3→32 channels, kernel=3×3, padding=1
- BatchNorm + ReLU + MaxPool(2×2)
- Conv Block 2: 32→64 channels, kernel=3×3, padding=1
- BatchNorm + ReLU + MaxPool(2×2)

**Classification Head**:
- Adaptive Average Pooling → 1×1 spatial dimension
- Flatten
- FC layer: 64 → 128
- ReLU + Dropout(0.5)
- FC layer: 128 → num_classes

### Model Statistics

| Parameter | Value |
|-----------|-------|
| Total Parameters | ~XXX,XXX |
| Trainable Parameters | ~XXX,XXX |
| Input Size | 224×224×3 |
| Output Size | num_classes |
| FLOPs | ~XXX M |
| Model Size | ~XX MB |

### Design Choices

1. **BatchNorm after Conv**: Improves training stability and convergence
2. **Adaptive Average Pooling**: Makes model flexible to different input sizes
3. **Dropout(0.5)**: Prevents overfitting in classification head
4. **ReLU activation**: Standard choice for efficiency

### Comparison with Other Architectures

| Model | Params | Accuracy | Inference Time |
|-------|--------|----------|----------------|
| ExampleCNN | XXX K | XX.X% | XX ms |
| ResNet18 | 11.2 M | XX.X% | XX ms |
| ResNet50 | 25.6 M | XX.X% | XX ms |

## Training Strategy

### Transfer Learning
For ResNet models, we use pretrained weights from ImageNet:
1. Load pretrained ResNet
2. Replace final FC layer for our num_classes
3. Option A: Freeze early layers, train only head
4. Option B: Fine-tune entire network with lower LR

### Loss Function
- **CrossEntropyLoss**: Standard for multi-class classification
- Combines LogSoftmax + NLLLoss

### Optimizer
- **Adam**: Learning rate = 0.001, weight_decay = 0.0001
- Alternative: SGD with momentum=0.9

### Learning Rate Schedule
- **ReduceLROnPlateau**: Reduce LR by 50% if val accuracy plateaus for 3 epochs
- Minimum LR: 1e-6

## References

- Add links to papers, blog posts, or documentation
- Related architectures
- Pretrained model sources
