import os


# Data directory
DATA_DIR = os.path.join("data")

# Raw and processed data directories
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
TOK_DIR = os.path.join(DATA_DIR, "tokenized")

# Model directory
MODEL_DIR = os.path.join("models")

# Ensure directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
