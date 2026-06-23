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
y  = subjects_df["label"].values
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
loaded = sum(1 for s in subjects_df.subject_id if s in all_epochs)
print(f"Loaded from cache: {loaded}/{len(subjects_df)}")
if loaded == 0:
    print("ERROR: No cached epochs found. Run eeg_pd_v2.py first to preprocess data.")
    sys.exit(1)
# Filter to subjects with cache
mask = subjects_df.subject_id.isin(all_epochs.keys())
subjects_df = subjects_df[mask].reset_index(drop=True)
y = subjects_df["label"].values

# STEP 3  —  EPOCH-LEVEL FEATURE EXTRACTIONSTEP
def get_ch_signal(ep, ch_names, fixed_ch, n_times):
    """Extract fixed channel signals from one epoch. Missing → zeros."""
    sigs = np.zeros((len(fixed_ch), n_times), dtype=np.float32)
    for fi, ch in enumerate(fixed_ch):
        if ch in ch_names:
            idx = ch_names.index(ch)
            sig = ep[idx]
            L   = min(len(sig), n_times)
            sigs[fi, :L] = sig[:L]
    return sigs

def wavelet_features_fast(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    feats  = []
    for c in coeffs[:6]:
        c = np.asarray(c, dtype=float)
        if len(c) < 2:
            feats.extend([0.0]*10); continue
        energy = float(np.sum(c**2))
        feats.extend([
            float(np.mean(np.abs(c))),      # abs mean
            float(np.std(c)),               # std
            energy,                          # energy
            float(np.sum(np.diff(np.sign(c)) != 0)) / max(len(c)-1,1),  # MCR
            float(np.mean(c[1:-1]**2 - c[:-2]*c[2:])) if len(c)>2 else 0.,  # Teager
             float(-np.sum((np.abs(c)/(np.sum(np.abs(c))+1e-10)) *  np.log2(np.abs(c)/(np.sum(np.abs(c))+1e-10)+1e-10))),  # entropy
            float(np.sqrt(energy/len(c))),  # RMS
            float(kurtosis(c)),
            float(skew(c)),
            float(np.median(c)),
        ])
    return feats 

def band_features(psd, freqs):
    total = np.trapezoid(psd, freqs) + 1e-10
    bps   = {}
    feats = []
    for band,(lo,hi) in BANDS.items():
        idx = np.logical_and(freqs>=lo, freqs<hi)   
        bp  = float(np.trapezoid(psd[idx], freqs[idx]))
        bps[band] = bp
        feats.extend([bp, bp/total])
 feats.extend([
        bps['alpha']/(bps['beta']+1e-10),
        bps['theta']/(bps['alpha']+1e-10),
        bps['delta']/(bps['theta']+1e-10),
        bps['alpha']+bps['beta'],           # total oscillatory
        bps['theta']/(bps['beta']+1e-10),   # theta/beta
    ])
    return feats  

def spectrogram_features(signal, sfreq, nperseg=64):
    """ Compute spectrogram and extract statistical features from
    time-frequency representation. Captures transient PD biomarkers. """
    f, t, Sxx = sig_spectrogram(signal, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    # Band-specific time-averaged power variance
    feats = []
    for lo,hi in [(0.5,4),(4,8),(8,13),(13,30),(30,45)]:
idx = np.logical_and(f>=lo, f<hi)
        if idx.sum() == 0:
            feats.extend([0.,0.,0.]); continue
        band_power_over_time = Sxx[idx,:].mean(axis=0)
        feats.extend([
            float(np.var(band_power_over_time)),      # temporal variance
            float(np.max(band_power_over_time)),      # peak
            float(np.mean(band_power_over_time)),     # mean
        ])
    return feats  

def bispectrum_feature(signal, sfreq):
    """ Simplified bispectrum: cross-biphase between alpha and beta.
    Full bispectrum achieves 99% in literature (arxiv 87-1). """
    ba, aa = butter(4, [8/(sfreq/2), 13/(sfreq/2)], btype='band')
    bb, ab = butter(4, [13/(sfreq/2), 30/(sfreq/2)], btype='band')
    s_a = filtfilt(ba, aa, signal)
    s_b = filtfilt(bb, ab, signal)
    h_a = hilbert(s_a); h_b = hilbert(s_b)
     # Biphase: phase of (x_alpha * x_alpha * conj(x_beta))
    biphase = np.angle(h_a * h_a * np.conj(h_b))
    return [float(np.mean(np.cos(biphase))), float(np.mean(np.sin(biphase))), float(np.std(biphase))]  # 3 features

def hjorth(sig):
    act = float(np.var(sig))
    d1  = np.diff(sig); d2 = np.diff(d1)
    mob = float(np.sqrt(np.var(d1)/(act+1e-10)))
    com = float(np.sqrt(np.var(d2)/(np.var(d1)+1e-10))/(mob+1e-10))
    return [act, mob, com]

def features_one_epoch(ep_sigs, sfreq, fixed_ch):
    """ ep_sigs: (n_fixed_ch, n_times)
    Returns feature vector for ONE epoch. """
    feats = []
    n_ch  = len(fixed_ch)
    for ci in range(n_ch):
        sig = ep_sigs[ci].astype(float)
        freqs, psd = welch(sig, fs=sfreq, nperseg=min(sfreq, len(sig)//2), noverlap=sfreq//4)

        feats.extend(wavelet_features_fast(sig))    # 60
        feats.extend(band_features(psd, freqs))     # 15
        feats.extend(spectrogram_features(sig, sfreq))  # 15
        feats.extend(bispectrum_feature(sig, sfreq))    # 3
        feats.extend(hjorth(sig))                   # 3
        feats.extend(perm_entropy(sig))             # 1
        # total per channel: 60+15+15+3+3+1 = 97

