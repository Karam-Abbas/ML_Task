# House Price Prediction

This project implements a machine learning pipeline for predicting house prices in Pakistan using DVC for data versioning and Flask for deployment.

## Structure

- `data/`: Contains the dataset
- `src/`: Source code for training and preprocessing
- `models/`: Trained models
- `app/`: Flask application
- `dvc.yaml`: DVC pipeline definition
- `params.yaml`: Configuration parameters
  
## Setup Steps
```python
# Create directories
mkdir -p data src models metrics app/templates

# Install dependencies
pip install dvc pandas scikit-learn flask joblib pyyaml

# Initialize DVC
dvc init

# Set up local remote storage
mkdir -p dvc_storage
dvc remote add -d local_storage dvc_storage

# Create .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "venv/" >> .gitignore
echo ".env" >> .gitignore
```

## To Update Model

```python 
# Pull latest data and model from DVC storage
dvc pull

# Run the pipeline
dvc repro

# Push updated model to DVC storage
dvc push

# Commit changes to Git
git add .
git commit -m "Update model with latest data"
git push origin main
```