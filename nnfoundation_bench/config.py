"""Project paths and benchmark-wide constants.

The data root can be moved off the repo (e.g. to a large scratch disk) with the
``NNFB_DATA_DIR`` environment variable; it must contain the CCDS sub-folders
``raw/``, ``interim/``, ``processed/`` and ``external/`` (created on demand).
"""

import os
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("NNFB_DATA_DIR", PROJ_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Third-party model weights (TotalSegmentator) live next to our own models.
TOTALSEG_HOME_DIR = Path(os.environ.get("TOTALSEG_HOME_DIR", MODELS_DIR / "totalsegmentator"))

# Classification protocol from nnFoundation (arXiv:2609.26924, Sec. 4 / Table 9):
# volumes resampled to 1 mm isotropic, 192^3 crop around the target region,
# 40/10/50 train/val/test split, AUROC.
TARGET_SPACING_MM = (1.0, 1.0, 1.0)
CROP_SIZE = (192, 192, 192)
SPLIT_FRACTIONS = {"train": 0.4, "val": 0.1, "test": 0.5}
DEFAULT_SEED = 12345

# Value used outside the field of view after cropping/resampling CT (air, in HU).
CT_PAD_VALUE = -1024
