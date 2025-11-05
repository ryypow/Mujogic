# Training Guide

## Quick Start

### 1. Prepare Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
# Organize dataset into train/val/test
python scripts/preprocessing/prepare_dataset.py \
    --input data/raw \
    --output data/processed
```

### 3. Train Model
```bash
# Using config file
python scripts/training/train_example.py \
    --config configs/example_config.yaml

# Or with command-line arguments
python scripts/training/train_example.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001
```

## Training from Scratch vs Fine-Tuning

### Training from Scratch
- Use when: You have large dataset (>10K images per class)
- Longer training time
- More GPU memory required
- Example: Custom CNN architectures

```bash
python scripts/training/train_baseline_cnn.py --epochs 100
```

### Fine-Tuning Pretrained Models
- Use when: Limited data (<10K total images)
- Faster convergence
- Better results with less data
- Example: ResNet18/50 with ImageNet weights

```bash
python scripts/training/train_resnet18.py --pretrained --epochs 30
```

## Hyperparameter Tuning

### Learning Rate
Start with these values:
- **From scratch**: 0.001 (Adam) or 0.01 (SGD)
- **Fine-tuning**: 0.0001 (10x lower)

Too high: Loss diverges or oscillates
Too low: Slow convergence

### Batch Size
- **Limited GPU memory**: 16-32
- **Moderate GPU (8-16GB)**: 64-128
- **Large GPU (24GB+)**: 256-512

Larger batch = more stable gradients, but may reduce generalization

### Number of Epochs
- Start with 30-50 epochs
- Use early stopping based on validation loss
- Monitor for overfitting (train acc >> val acc)

## Monitoring Training

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir results/

# View at http://localhost:6006
```

### What to Watch
1. **Training/Validation Loss**: Should both decrease
2. **Training/Validation Accuracy**: Should both increase
3. **Gap between train and val**: Large gap = overfitting

### Signs of Problems

**Overfitting**:
- Symptoms: Train acc high, val acc low
- Solutions: Add dropout, reduce model size, data augmentation

**Underfitting**:
- Symptoms: Both train and val acc low
- Solutions: Increase model capacity, train longer, reduce regularization

**Learning Rate Too High**:
- Symptoms: Loss oscillates or diverges
- Solution: Reduce learning rate by 10x

**Learning Rate Too Low**:
- Symptoms: Loss decreases very slowly
- Solution: Increase learning rate

## Data Augmentation

### Training Augmentations
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

### Validation/Test Augmentations
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## Experiment Tracking

### Naming Convention
```
{model}_{date}_{experiment_id}
```

Example: `resnet18_20241024_exp001`

### What to Track
- Model architecture
- Hyperparameters (LR, batch size, epochs)
- Training/validation metrics
- Final test accuracy
- Training time
- Hardware used

### Example Experiment Log
```yaml
experiment_id: exp001
date: 2024-10-24
model: resnet18
dataset: kitchen-utensils
hyperparameters:
  batch_size: 64
  learning_rate: 0.0001
  epochs: 30
  optimizer: adam
results:
  train_acc: 0.985
  val_acc: 0.932
  test_acc: 0.928
  training_time: "45 minutes"
  gpu: "NVIDIA RTX 3090"
notes: "Baseline experiment with default hyperparameters"
```

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
python train.py --batch-size 16  # instead of 32

# Use gradient accumulation
# Accumulate gradients over N batches before updating
```

### Slow Training
- Increase num_workers in DataLoader
- Use mixed precision training (fp16)
- Move data preprocessing to GPU
- Use faster data augmentation library (albumentations)

### Poor Convergence
- Check learning rate (try 10x higher/lower)
- Verify data normalization is correct
- Check for class imbalance
- Ensure proper train/val split

## Best Practices

1. **Always use a validation set** - Don't touch test set until final evaluation
2. **Save checkpoints** - Save best model based on val accuracy
3. **Use reproducible seeds** - Set random seed for reproducibility
4. **Monitor GPU utilization** - Use `nvidia-smi` to check GPU usage
5. **Start simple** - Begin with small model, then scale up
6. **Log everything** - Track all experiments in a spreadsheet or W&B
7. **Version your code** - Use git to track code changes

## Multi-GPU Training

### DataParallel (Simple)
```python
model = nn.DataParallel(model)
```

### DistributedDataParallel (Faster)
```bash
python -m torch.distributed.launch --nproc_per_node=2 train.py
```

## Additional Resources

- [PyTorch Training Guide](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [How to Train Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
- [Deep Learning Book](http://www.deeplearningbook.org/)
