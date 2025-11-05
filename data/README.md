# Dataset Documentation

## Overview
Describe your dataset here.

## Source
- **Name**: Dataset Name
- **URL**: https://dataset-source.com
- **License**: License type
- **Citation**:
  ```
  @article{author2024,
    title={Dataset Title},
    author={Author Name},
    year={2024}
  }
  ```

## Dataset Statistics

| Split | Samples | Classes | Size |
|-------|---------|---------|------|
| Train | X,XXX   | XX      | XXX MB |
| Val   | X,XXX   | XX      | XXX MB |
| Test  | X,XXX   | XX      | XXX MB |
| **Total** | **X,XXX** | **XX** | **XXX MB** |

## Class Distribution

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| class1 | XXX | XX | XX | XXX |
| class2 | XXX | XX | XX | XXX |
| ... | ... | ... | ... | ... |

## Directory Structure

```
data/
├── raw/                    # Original unprocessed data
│   └── [downloaded files]
└── processed/              # Cleaned and organized data
    ├── train/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    ├── val/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── test/
        ├── class1/
        ├── class2/
        └── ...
```

## How to Obtain Dataset

### Option 1: Direct Download
```bash
# Download from source
wget [DATASET_URL]
unzip dataset.zip -d data/raw/
```

### Option 2: Manual Download
1. Visit [DATASET_URL]
2. Download the dataset
3. Extract to `data/raw/`

### Option 3: Kaggle API
```bash
kaggle datasets download -d [username/dataset-name] -p data/raw/
unzip data/raw/dataset-name.zip -d data/raw/
```

## Preprocessing Steps

1. **Data cleaning**: Remove corrupted images
2. **Resizing**: Resize all images to 224x224
3. **Normalization**: Apply ImageNet normalization
4. **Splitting**: Split into train (70%), val (15%), test (15%)
5. **Augmentation**: Apply augmentations to training set

Run preprocessing:
```bash
python scripts/preprocessing/prepare_dataset.py --input data/raw --output data/processed
```

## Image Specifications

- **Format**: JPEG/PNG
- **Size**: 224x224 pixels
- **Channels**: RGB (3 channels)
- **Normalization**: ImageNet stats
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

## Known Issues

- List any known issues with the dataset
- Missing classes
- Class imbalance
- Corrupted files

## References

- Link to dataset paper/documentation
- Related datasets
- Benchmark results
