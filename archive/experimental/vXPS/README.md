# Machine Learning Project Template

A professional, production-ready template for PyTorch/ML projects following industry best practices.

---

## 🏗️ Project Structure

```
project-name/
├── README.md                          # This file - project documentation
├── LICENSE                            # Project license (MIT/Apache 2.0)
├── requirements.txt                   # Python dependencies
├── setup.py                           # Make project pip installable
├── .gitignore                         # Git ignore rules
│
├── src/                               # 📦 LIBRARY CODE (importable modules)
│   ├── __init__.py
│   ├── models/                        # Model architectures
│   │   ├── __init__.py
│   │   └── your_model.py              # Model class definitions
│   ├── data/                          # Dataset classes
│   │   ├── __init__.py
│   │   └── dataset.py                 # PyTorch Dataset/DataLoader classes
│   └── utils/                         # Helper functions
│       ├── __init__.py
│       ├── metrics.py                 # Evaluation metrics
│       └── visualization.py           # Plotting utilities
│
├── scripts/                           # 🚀 EXECUTABLE SCRIPTS (CLI entry points)
│   ├── training/                      # Training scripts
│   │   ├── train_model1.py            # python scripts/training/train_model1.py
│   │   └── train_model2.py
│   ├── evaluation/                    # Evaluation scripts
│   │   ├── evaluate.py                # Model evaluation
│   │   └── inference.py               # Single prediction/batch inference
│   └── preprocessing/                 # Data preprocessing
│       ├── prepare_dataset.py
│       └── augment_data.py
│
├── configs/                           # ⚙️ CONFIGURATION FILES
│   ├── model1_config.yaml             # Hyperparameters, paths, settings
│   └── model2_config.yaml
│
├── data/                              # 📊 DATASETS (git-ignored)
│   ├── README.md                      # Dataset documentation
│   ├── raw/                           # Original, unprocessed data
│   └── processed/                     # Cleaned, preprocessed data
│       ├── train/
│       ├── val/
│       └── test/
│
├── notebooks/                         # 📓 JUPYTER NOTEBOOKS
│   ├── 01_data_exploration.ipynb      # EDA and visualization
│   ├── 02_model_experiments.ipynb     # Interactive model testing
│   └── 03_results_analysis.ipynb      # Results visualization
│
├── models/                            # 💾 TRAINED MODEL WEIGHTS (git-ignored)
│   ├── model_name/
│   │   ├── best_model.pth
│   │   └── final_model.pth
│   └── checkpoints/                   # Training checkpoints
│
├── results/                           # 📈 EXPERIMENT OUTPUTS (git-ignored)
│   ├── model_name/
│   │   ├── metrics.json               # Numerical results
│   │   ├── confusion_matrix.png       # Visualizations
│   │   └── tensorboard/               # TensorBoard logs
│   └── experiments.csv                # Summary of all experiments
│
├── tests/                             # 🧪 UNIT TESTS
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_dataset.py
│   └── test_training.py
│
├── docs/                              # 📚 ADDITIONAL DOCUMENTATION
│   ├── ARCHITECTURE.md                # Model architecture details
│   ├── DATASET.md                     # Dataset description
│   └── TRAINING.md                    # Training procedures
│
└── .github/                           # GitHub-specific files
    └── workflows/
        └── ci.yml                     # GitHub Actions for CI/CD
```

---

## 📋 Directory Descriptions

### **src/** - Library Code (Importable)
Code that you **import** in other scripts. Contains reusable components.

- **models/** - Model architecture definitions (PyTorch nn.Module classes)
- **data/** - Dataset and DataLoader implementations
- **utils/** - Helper functions, metrics, visualization utilities

**Usage:**
```python
from src.models import MyModel
from src.data import MyDataset
from src.utils.metrics import calculate_accuracy
```

---

### **scripts/** - Executable Scripts
Code that you **run directly** from the command line.

- **training/** - Scripts to train models (`python scripts/training/train_model.py`)
- **evaluation/** - Scripts to evaluate trained models
- **preprocessing/** - Scripts to prepare/clean/augment data

**Usage:**
```bash
python scripts/training/train_resnet.py --epochs 50 --batch-size 32
python scripts/evaluation/evaluate.py --model models/resnet/best_model.pth
python scripts/preprocessing/augment_data.py --input data/raw --output data/processed
```

---

### **configs/** - Configuration Files
YAML or JSON files containing hyperparameters, paths, and settings.

**Example config.yaml:**
```yaml
model:
  name: resnet18
  num_classes: 10
  pretrained: true

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam

data:
  train_path: data/processed/train
  val_path: data/processed/val
  test_path: data/processed/test
```

---

### **data/** - Datasets (Git-Ignored)
Store your datasets here. **Never commit large data files to git.**

- **raw/** - Original data exactly as downloaded
- **processed/** - Cleaned, augmented, or transformed data

**Include data/README.md** with:
- Dataset source and download instructions
- Dataset statistics (size, classes, splits)
- Preprocessing steps applied

---

### **notebooks/** - Jupyter Notebooks
Interactive exploration, experimentation, and visualization.

**Recommended notebooks:**
1. **01_data_exploration.ipynb** - EDA, class distribution, sample visualization
2. **02_model_experiments.ipynb** - Quick model prototyping
3. **03_results_analysis.ipynb** - Compare models, plot metrics

---

### **models/** - Trained Weights (Git-Ignored)
Store trained model checkpoints. **Never commit .pth files to git** (too large).

Use Git LFS for Hugging Face or cloud storage (AWS S3, Google Cloud Storage).

---

### **results/** - Experiment Outputs (Git-Ignored)
Store training logs, metrics, and visualizations.

**Typical contents:**
- `metrics.json` - Accuracy, loss, F1, etc.
- `confusion_matrix.png` - Evaluation plots
- `tensorboard/` - TensorBoard logs
- `experiments.csv` - Summary table of all runs

---

### **tests/** - Unit Tests
Automated tests for CI/CD pipelines.

```bash
pytest tests/
```

---

## 🚀 Recommended Workflow

### **Phase 1: Setup** (One-time)
```bash
# 1. Clone template
git clone <template-repo> my-new-project
cd my-new-project

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize git
git init
git add .
git commit -m "Initial commit from ML template"
```

---

### **Phase 2: Data Preparation**
```bash
# 1. Add raw data to data/raw/
#    (Download or copy your dataset)

# 2. Document the dataset
#    Edit data/README.md with source, stats, license

# 3. Explore data
jupyter notebook notebooks/01_data_exploration.ipynb

# 4. Preprocess data
python scripts/preprocessing/prepare_dataset.py

# Result: Clean data in data/processed/
```

---

### **Phase 3: Model Development**

#### **A. Define Model Architecture**
```bash
# 1. Create model class in src/models/
#    Example: src/models/my_cnn.py

# 2. Create dataset class in src/data/
#    Example: src/data/my_dataset.py

# 3. Test in notebook
#    Use notebooks/02_model_experiments.ipynb for quick prototyping
```

#### **B. Create Training Script**
```bash
# 1. Create scripts/training/train_my_model.py
#    - Load data
#    - Initialize model
#    - Training loop
#    - Save checkpoints

# 2. Create config file
#    configs/my_model_config.yaml

# 3. Run training
python scripts/training/train_my_model.py --config configs/my_model_config.yaml
```

---

### **Phase 4: Training & Experimentation**
```bash
# 1. Train baseline model
python scripts/training/train_baseline.py --epochs 50

# 2. Monitor with TensorBoard
tensorboard --logdir results/baseline/tensorboard

# 3. Try different configurations
python scripts/training/train_resnet.py --config configs/resnet18.yaml
python scripts/training/train_resnet.py --config configs/resnet50.yaml

# 4. Track experiments
#    Results automatically saved to results/
```

---

### **Phase 5: Evaluation**
```bash
# 1. Evaluate on test set
python scripts/evaluation/evaluate.py \
    --model models/resnet18/best_model.pth \
    --data data/processed/test

# 2. Run inference on new images
python scripts/evaluation/inference.py \
    --model models/resnet18/best_model.pth \
    --image sample.jpg

# 3. Visualize results
jupyter notebook notebooks/03_results_analysis.ipynb
```

---

### **Phase 6: Documentation & Sharing**
```bash
# 1. Update README.md with:
#    - Project description
#    - Installation instructions
#    - Usage examples
#    - Results/benchmarks

# 2. Write additional docs
#    - docs/ARCHITECTURE.md - Model details
#    - docs/DATASET.md - Data information
#    - docs/TRAINING.md - Training procedures

# 3. Clean up before committing
git status  # Check what's tracked
# Ensure data/, models/, results/ are in .gitignore

# 4. Commit and push
git add .
git commit -m "Add model architecture and training pipeline"
git push origin main
```

---

## 📦 Making Your Project Pip-Installable

Create `setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name="your-project-name",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        # Add other dependencies
    ],
)
```

**Install in development mode:**
```bash
pip install -e .
```

**Now you can import from anywhere:**
```python
from src.models import MyModel  # Works from any directory!
```

---

## 🎯 Best Practices

### **1. Use .gitignore**
Never commit:
- Large data files (`data/`)
- Model weights (`models/`, `*.pth`)
- Virtual environments (`.venv/`, `venv/`)
- Generated outputs (`results/`, `__pycache__/`)

### **2. Separate Code from Configuration**
- Hardcode nothing (use config files)
- Makes experiments reproducible
- Easy to share hyperparameters

### **3. Version Control Everything Except Data**
- ✅ Commit: Code, configs, notebooks, docs
- ❌ Don't commit: Data, models, results

### **4. Document As You Go**
- Update README.md with each milestone
- Add docstrings to functions
- Comment complex logic

### **5. Test Your Code**
```bash
pytest tests/  # Run before committing
```

### **6. Use Meaningful Names**
```bash
# Good
scripts/training/train_resnet18_imagenet.py
models/resnet18_pretrained_epoch50.pth
results/resnet18_20241024_experiment1/

# Bad
scripts/train.py
models/model.pth
results/output/
```

---

## 🔄 Git Workflow

### **Initial Setup**
```bash
git init
git add .
git commit -m "Initial project structure from template"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

### **Daily Development**
```bash
# 1. Create feature branch
git checkout -b feature/new-model

# 2. Make changes, test
python scripts/training/train_new_model.py

# 3. Commit changes
git add src/models/new_model.py scripts/training/train_new_model.py
git commit -m "Add new CNN architecture"

# 4. Push and create PR
git push origin feature/new-model
# Create Pull Request on GitHub
```

---

## 📤 Uploading to Hugging Face

### **Option 1: Model Hub**
```bash
# Install huggingface_hub
pip install huggingface_hub

# Upload model
huggingface-cli login
huggingface-cli upload your-username/model-name models/best_model.pth
```

### **Option 2: Spaces (Demo)**
Create `app.py`:
```python
import gradio as gr
import torch
from src.models import MyModel

model = torch.load("models/best_model.pth")

def predict(image):
    # Inference logic
    return predictions

demo = gr.Interface(fn=predict, inputs="image", outputs="label")
demo.launch()
```

---

## 🎓 Learning Resources

- **PyTorch Project Structure**: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- **Python Packaging**: https://packaging.python.org/tutorials/packaging-projects/
- **Git Best Practices**: https://www.conventionalcommits.org/
- **ML Experiment Tracking**: https://www.tensorflow.org/tensorboard
- **Testing**: https://docs.pytest.org/

---

## ❓ FAQ

**Q: Should scripts import from src or copy code?**
A: Always import from src. Don't duplicate code.

**Q: Where do I put config files - in scripts or configs/?**
A: Keep them in `configs/` and reference from scripts.

**Q: Should I commit Jupyter notebooks?**
A: Yes, but clear outputs first (`jupyter nbconvert --clear-output`).

**Q: How do I handle large datasets?**
A: Never commit to git. Use Git LFS, cloud storage, or document download instructions.

**Q: One train.py or multiple training scripts?**
A: For <5 models: separate scripts. For 10+ models: unified train.py with configs.

---

## 📝 Customization Checklist

- [ ] Rename project in README.md
- [ ] Update requirements.txt with your dependencies
- [ ] Add LICENSE file (MIT/Apache 2.0)
- [ ] Update .gitignore for your specific needs
- [ ] Create src/models/ with your architectures
- [ ] Create src/data/ with your dataset classes
- [ ] Create training scripts in scripts/training/
- [ ] Create config files in configs/
- [ ] Document dataset in data/README.md
- [ ] Create initial notebooks for exploration
- [ ] Set up Git repository
- [ ] (Optional) Add CI/CD with GitHub Actions

---

## 📄 License

This template is provided under the MIT License. Feel free to use for any project.

---

**Happy ML Engineering! 🚀**
