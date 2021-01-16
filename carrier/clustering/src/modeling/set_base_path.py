import sys
from pathlib2 import Path

root_folder = Path(__file__).resolve().parents[2]
if root_folder not in sys.path:
    sys.path.insert(0, str(root_folder))
