# %%

import sys
from pathlib2 import Path
root_path = Path(__file__).resolve().parents[1]

if root_path not in sys.path:
    sys.path.insert(0, str(root_path))

DATA_BASE_PATH = root_path / 'data'
INTERIM_DATA_PATH = DATA_BASE_PATH / 'interim'
PROCESSED_DATA_PATH = DATA_BASE_PATH / 'processed'
RAW_DATA_PATH = DATA_BASE_PATH / 'raw'

MODEL_BASE_PATH = root_path / 'model'
ENC_MODEL_PATH = MODEL_BASE_PATH / 'encoded'
DIMRED_MODEL_PATH = MODEL_BASE_PATH / 'dimred'
