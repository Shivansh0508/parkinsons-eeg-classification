import os
import sys
os.environ["PYTHONUTF8"] = "1"
import ssl
import urllib3

# Disable SSL verification globally — nilearn atlas downloads fail
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch requests to skip SSL verify
import requests
from requests.adapters import HTTPAdapter
_orig_request = requests.Session.request
def _no_ssl_request(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig_request(self, method, url, **kwargs)
requests.Session.request = _no_ssl_request

import glob
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import nibabel as nib
from nilearn import image as nl_image
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn.image import smooth_img
from nilearn.datasets import (load_mni152_template, load_mni152_brain_mask, fetch_atlas_aal, fetch_atlas_harvard_oxford)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, f1_score, roc_curve, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ===========================================================================
# CONFIG
# ===========================================================================

CONFIG = {
    "HP_DIR"   : r"C:\Users\Shivansh\Downloads\HP",
    "PD_DIR"   : r"C:\Users\Shivansh\Downloads\PD",
    "CACHE_DIR": r"C:\Users\Shivansh\Downloads\BrainLat_Cache",
    "OUT_DIR"  : r"C:\Users\Shivansh\Downloads\BrainLat_Results",
    "N_FOLDS"  : 5,
    "SEED"     : 42,
}

for d in [CONFIG["CACHE_DIR"], CONFIG["OUT_DIR"]]:
    os.makedirs(d, exist_ok=True)

hp_files = (glob.glob(os.path.join(CONFIG["HP_DIR"], "*.nii.gz")) +
            glob.glob(os.path.join(CONFIG["HP_DIR"], "*.nii")))

pd_files = (glob.glob(os.path.join(CONFIG["PD_DIR"], "*.nii.gz")) +
            glob.glob(os.path.join(CONFIG["PD_DIR"], "*.nii")))

print(f"HP: {len(hp_files)}  |  PD: {len(pd_files)}  |  Total: {len(hp_files)+len(pd_files)}")

if not hp_files:
    raise FileNotFoundError(f"No MRI files found in {CONFIG['HP_DIR']}")
if not pd_files:
    raise FileNotFoundError(f"No MRI files found in {CONFIG['PD_DIR']}")

# SUBJECT TABLE
# Each subject appears exactly once. Site is extracted from the subject ID
ef build_table(hp_dir, pd_dir):
    rows = []
    for label, folder in [(1, pd_dir), (0, hp_dir)]:
        for path in sorted(glob.glob(os.path.join(folder, "*.nii.gz")) +
                           glob.glob(os.path.join(folder, "*.nii"))):
            fname = os.path.basename(path)
            sid   = (fname.replace("_T1w.nii.gz", "")
                         .replace("_T1w.nii", "")
                         .replace(".nii.gz", "")
                         .replace(".nii", ""))
            site  = ''.join(c for c in sid.replace("sub-", "") if c.isalpha())
            rows.append(dict(subject_id=sid, label=label, site=site, path=path))
                               
df = pd.DataFrame(rows).reset_index(drop=True)

    # Identify duplicates and print them so you can inspect
    dupes = df[df.duplicated(subset="subject_id", keep=False)]
    if len(dupes) > 0:
        print(f"\nWARNING: {dupes['subject_id'].nunique()} subject ID(s) appear in both folders:")
        print(dupes[["subject_id", "label", "path"]].to_string())
        print()
        
        # A subject in both folders means it was placed in both HP and PD by mistake.
        # Resolution: keep the PD label (label=1) as the ground truth, drop the HC duplicate.
        # Sort so PD rows (label=1) come first, then drop duplicates keeping first occurrence.
        df = df.sort_values("label", ascending=False).drop_duplicates(
            subset="subject_id", keep="first").reset_index(drop=True)
        print(f"Resolved: kept PD label for duplicated subjects.")

    print(f"Subjects : {len(df)}  |  PD: {(df.label==1).sum()}  |  HC: {(df.label==0).sum()}")
    print(f"Sites    : {sorted(df.site.unique())}")
    print(f"Subjects per site:\n{df.groupby(['site','label']).size().unstack(fill_value=0)}")
    return df


subjects_df = build_table(CONFIG["HP_DIR"], CONFIG["PD_DIR"])

 # PREPROCESSING
# All steps are deterministic transforms applied per-subject independently.
# No statistics are shared across subjects during preprocessing.
# Z-score is computed within each subject's own brain mask voxels only —
# this is NOT a population-level normalisation and introduces no leakage.
# Cache ensures reproducibility.

try:
    import ants
    ANTS_OK = True
    print("ANTs available — N4 bias correction + affine MNI registration enabled")
except ImportError:
    ANTS_OK = False
    print("ANTs not found — using nilearn resample fallback")

_MNI, _MASK = None, None

def get_mni():
    global _MNI, _MASK
    if _MNI is None:
        _MNI  = load_mni152_template(resolution=2)
        _MASK = load_mni152_brain_mask(resolution=2)
    return _MNI, _MASK
