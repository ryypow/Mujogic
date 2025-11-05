# Quick Usage Guide

## One-Time Setup

```bash
# 1. Clone/copy this template
cp -r ml-project-template my-new-project
cd my-new-project

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make pip-installable (optional)
pip install -e .
```

## Typical Workflow

### 1. Add Your Data
```bash
# Place raw data in data/raw/
# Document it in data/README.md
```

### 2. Create Your Model
```bash
# Add model class to src/models/your_model.py
# Example: src/models/my_cnn.py
```

### 3. Create Training Script
```bash
# Copy template: scripts/training/train_example.py
# Modify for your model: scripts/training/train_my_model.py
```

### 4. Train
```bash
python scripts/training/train_my_model.py --epochs 50 --batch-size 32
```

### 5. Evaluate
```bash
python scripts/evaluation/evaluate.py --model models/my_model/best_model.pth
```

### 6. Run Inference
```bash
python scripts/evaluation/inference.py --model models/my_model/best_model.pth --image test.jpg
```

## File Organization

- **Write code in src/** - For reusable components
- **Write scripts in scripts/** - For executable programs
- **Put configs in configs/** - For hyperparameters
- **Never commit data or models** - They're git-ignored

## Getting Help

See detailed documentation in:
- `README.md` - Full documentation
- `docs/ARCHITECTURE.md` - Model details
- `docs/TRAINING.md` - Training guide
- `data/README.md` - Dataset info
