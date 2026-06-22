# EEG PD vs HC  —  DS007526 
import os, sys, ssl, glob, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
os.environ["PYTHONUTF8"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context
def check():
    missing = []
    for pkg, imp in [("mne","mne"),("eegdash","eegdash"),("sklearn","sklearn"),
                     ("xgboost","xgboost"),("lightgbm","lightgbm"),
                     ("imbalanced-learn","imblearn"),("scipy","scipy"),
                     ("PyWavelets","pywt"),("torch","torch"),
                     ("matplotlib","matplotlib"),("seaborn","seaborn")]:
        try: __import__(imp)
        except ImportError: missing.append(pkg)
    if missing:
        print("Run:  pip install " + " ".join(missing))
        sys.exit(1)
    print("All packages OK")
check()
import mne; mne.set_log_level('WARNING')
import pywt, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from eegdash import EEGDash
from scipy.signal import welch, butter, filtfilt, hilbert, spectrogram as sig_spectrogram
from scipy.stats import kurtosis, skew
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix, f1_score, precision_score, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb, lightgbm as lgb
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import seaborn as sns
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"MNE {mne.__version__}  Python {sys.version[:6]}  Device: {DEVICE}")

# CONFIG
CONFIG = {
    "CACHE_DIR" : r"C:\Users\Downloads\PD_EEG_v2",  
    "OUT_DIR"   : r"C:\Users\Downloads\EEG_Results",
    "SFREQ"     : 250,
    "L_FREQ"    : 0.5,
    "H_FREQ"    : 45.0,
    "NOTCH"     : 50.0,
    "EPOCH_LEN" : 4.0,
    "OVERLAP"   : 0.5,
    "AMP_THRESH": 150e-6,
    "N_FOLDS"   : 5,
    "SEED"      : 42,
}
os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
FIXED_CH = ['F3','Fz','F4','T7','C3','C4','P7','P3','Pz','P4','P8','O1','Oz','O2']
N_CH     = len(FIXED_CH)  
SFREQ    = CONFIG["SFREQ"]
N_TIMES  = int(CONFIG["EPOCH_LEN"] * SFREQ)  

BANDS = {'delta':(0.5,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,45)}

# STEP 1  —  METADATA
print("\nQuerying EEGDash...")
client      = EEGDash()
all_records = client.find(dataset="ds007526")
rows = []
seen = set()
for rec in all_records:
    if rec.get("task","") != "rest": continue
    pinfo    = rec.get("participant_tsv", {}) or {}
    sid      = pinfo.get("subject_id")
    if not sid or sid in seen: continue
    seen.add(sid)
    bids_sub = str(rec.get("subject","")).zfill(3)
    grp      = str(pinfo.get("group","")).upper()
    rows.append(dict(subject_id=sid, bids_sub=bids_sub,label=0 if grp=="HC" else 1, record=rec))

subjects_df = pd.DataFrame(rows).reset_index(drop=True)
y           = subjects_df["label"].values
print(f"Subjects: {len(subjects_df)}  PD={y.sum()}  HC={len(y)-y.sum()}")

# STEP 2  —  LOAD PREPROCESSED EPOCHS FROM V2 CACHE
all_epochs   = {}
all_channels = {}
for _, row in subjects_df.iterrows():
    sid   = row["subject_id"]
    ep_p  = os.path.join(CONFIG["CACHE_DIR"], f"{sid}_ep.npy")
    ch_p  = os.path.join(CONFIG["CACHE_DIR"], f"{sid}_ch.npy")
    if os.path.exists(ep_p) and os.path.exists(ch_p):
        all_epochs[sid]   = np.load(ep_p)
        all_channels[sid] = list(np.load(ch_p, allow_pickle=True))
    else:
        print(f"  WARNING: no cache for {sid} — run v2 first to preprocess")
