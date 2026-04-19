# src/config.py

from pathlib import Path

# Dataset settings
NUM_EMPLOYEES = 1000
RANDOM_SEED = 42

# File paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "employee_performance_data.csv"

# Departments
DEPARTMENTS = ["IT", "HR", "Sales", "Finance", "Operations"]

# Education levels
EDUCATION_LEVELS = ["Bachelor", "Master", "PhD"]

# Performance thresholds
HIGH_PERFORMANCE_THRESHOLD = 0.75
MEDIUM_PERFORMANCE_THRESHOLD = 0.50