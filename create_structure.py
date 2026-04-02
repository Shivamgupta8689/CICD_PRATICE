import os

# Folder structure
folders = [
    "mlops-churn-project/data/raw",
    "mlops-churn-project/data/processed",
    "mlops-churn-project/src",
    "mlops-churn-project/pipeline",
    "mlops-churn-project/app",
    "mlops-churn-project/models",
    "mlops-churn-project/.github/workflows"
]

# Files to create
files = [
    "mlops-churn-project/src/data_ingestion.py",
    "mlops-churn-project/src/data_preprocessing.py",
    "mlops-churn-project/src/train.py",
    "mlops-churn-project/src/evaluate.py",
    "mlops-churn-project/pipeline/training_pipeline.py",
    "mlops-churn-project/app/app.py",
    "mlops-churn-project/dvc.yaml",
    "mlops-churn-project/params.yaml",
    "mlops-churn-project/requirements.txt",
    "mlops-churn-project/Dockerfile",
    "mlops-churn-project/.github/workflows/mlops.yml",
    "mlops-churn-project/README.md"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file in files:
    with open(file, "w") as f:
        f.write("")

print("✅ MLOps project structure created successfully!")