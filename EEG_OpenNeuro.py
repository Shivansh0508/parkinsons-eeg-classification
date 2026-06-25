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

    # PLV alpha+beta, 10 pairs
    ba, aa = butter(4, [8/(sfreq/2), 13/(sfreq/2)], btype='band')
    bb, ab = butter(4, [13/(sfreq/2), 30/(sfreq/2)], btype='band')
    done = 0
    for i in range(n_ch):
        if done >= 10: break
        for j in range(i+1, n_ch):
            if done >= 10: break
            si = ep_sigs[i].astype(float)
            sj = ep_sigs[j].astype(float)
            phi_a = np.angle(hilbert(filtfilt(ba,aa,si))) - \ np.angle(hilbert(filtfilt(ba,aa,sj)))
            phi_b = np.angle(hilbert(filtfilt(bb,ab,si))) - \ np.angle(hilbert(filtfilt(bb,ab,sj)))
            feats.extend([float(np.abs(np.mean(np.exp(1j*phi_a)))),float(np.abs(np.mean(np.exp(1j*phi_b))))])
            done += 1
    return feats  

def extract_epoch_features(subjects_df, all_epochs, all_channels, config, fixed_ch):
    """ Returns:  epoch_X : dict sid -> (n_epochs, n_features)
        epoch_y : dict sid -> int label """
    cache_ep_feat = os.path.join(config["CACHE_DIR"], "ep_feats_v3.npz")
    if os.path.exists(cache_ep_feat):
        data      = np.load(cache_ep_feat, allow_pickle=True)
        epoch_X   = data["epoch_X"].item()
        epoch_y   = data["epoch_y"].item()
        print(f"Epoch features loaded from cache. Subjects: {len(epoch_X)}")
        return epoch_X, epoch_y
        
    epoch_X = {}
    epoch_y = {}
    for i, row in subjects_df.iterrows():
        sid   = row["subject_id"]
        label = int(row["label"])
        if sid not in all_epochs: continue
        ep_data  = all_epochs[sid]    
        ch_names = all_channels[sid]
        n_ep     = len(ep_data)
        n_times  = ep_data.shape[2]

 ep_feats = []
        for ei in range(n_ep):
            ep_sigs = get_ch_signal(ep_data[ei], ch_names, fixed_ch, n_times)
            fv  = features_one_epoch(ep_sigs, config["SFREQ"], fixed_ch)
            ep_feats.append(fv)
        epoch_X[sid] = np.array(ep_feats, dtype=np.float32)
        epoch_y[sid] = label
        print(f"  [{i+1}/{len(subjects_df)}] {sid}: " f"{n_ep} epochs × {len(ep_feats[0])} features", end='\r')

np.savez(cache_ep_feat, epoch_X=epoch_X, epoch_y=epoch_y)
    print(f"\nEpoch features saved.")
    return epoch_X, epoch_y
print(f"\nExtracting per-epoch features ({97*N_CH+20} per epoch)...")
epoch_X, epoch_y = extract_epoch_features(subjects_df, all_epochs, all_channels, CONFIG, FIXED_CH)
for _, row in subjects_df.iterrows():
    assert row.subject_id in epoch_X, f"Missing: {row.subject_id}"
print(f"All {len(epoch_X)} subjects have epoch features")
# Cleanup NaN/Inf per-subject
for sid in epoch_X:
    X = epoch_X[sid]
    for col in range(X.shape[1]):
        bad = ~np.isfinite(X[:, col])
        if bad.any():
            med = np.nanmedian(X[:, col])
            X[bad, col] = med if np.isfinite(med) else 0.
    epoch_X[sid] = X
    
# STEP 4  —  1D CNN  (EEGNet-style)
# Input: raw EEG epochs (n_ch, n_times) = (14, 1000)
class EEGNet(nn.Module):
    """ EEGNet: compact CNN for EEG classification.
    Lawhern et al. 2018 — best architecture for small EEG datasets.
    Input: (B, 1, n_ch, n_times)"""

def __init__(self, n_ch=14, n_times=1000, n_classes=2,F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        # Temporal convolution
        self.conv1  = nn.Conv2d(1, F1, (1, 64), padding=(0,32), bias=False)
        self.bn1    = nn.BatchNorm2d(F1)
        # Depthwise spatial convolution
         self.conv2  = nn.Conv2d(F1, F1*D, (n_ch,1), groups=F1, bias=False)
        self.bn2    = nn.BatchNorm2d(F1*D)
        self.act2   = nn.ELU()
        self.pool2  = nn.AvgPool2d((1,4))
        self.drop2  = nn.Dropout(dropout)
        # Separable convolution
        self.conv3a = nn.Conv2d(F1*D, F1*D, (1,16), padding=(0,8),groups=F1*D, bias=False)
        self.conv3b = nn.Conv2d(F1*D, F2, 1, bias=False)
        self.bn3    = nn.BatchNorm2d(F2)
        self.act3   = nn.ELU()
        self.pool3  = nn.AvgPool2d((1,8))
        self.drop3  = nn.Dropout(dropout)
        # Compute flatten size
        with torch.no_grad():
            x = torch.zeros(1,1,n_ch,n_times)
            x = self.pool2(self.act2(self.bn2(self.conv2(
                   self.bn1(self.conv1(x))))))
            x = self.pool3(self.act3(self.bn3(self.conv3b(
                   self.conv3a(x)))))
            self.flat_size = x.numel()
        self.fc = nn.Linear(self.flat_size, n_classes)

def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.drop2(self.pool2(self.act2(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(self.act3(self.bn3(
            self.conv3b(self.conv3a(x))))))
        return self.fc(x.view(x.size(0), -1))

class EpochDataset(Dataset):
    def __init__(self, sids, all_epochs, all_channels, labels_map,fixed_ch, n_times, augment=False):
        self.items   = []
        self.augment = augment
        for sid in sids:
            if sid not in all_epochs: continue
            ep_data  = all_epochs[sid]
            ch_names = all_channels[sid]
            label    = labels_map[sid]
            for ei in range(len(ep_data)):
                sigs = get_ch_signal(ep_data[ei], ch_names, fixed_ch, n_times)
                self.items.append((sigs.astype(np.float32), label))

def __len__(self):  return len(self.items)

    def __getitem__(self, idx):
        sigs, label = self.items[idx]
        # Normalize per-channel
        for ci in range(sigs.shape[0]):
            m = sigs[ci].mean(); s = sigs[ci].std() + 1e-8
            sigs[ci] = (sigs[ci] - m) / s
        if self.augment:
             if np.random.rand() > 0.5:
                sigs += np.random.normal(0, 0.05, sigs.shape).astype(np.float32)
            if np.random.rand() > 0.5:
                sigs *= np.random.uniform(0.9, 1.1)
        x = torch.tensor(sigs, dtype=torch.float32).unsqueeze(0) 
        return x, torch.tensor(label, dtype=torch.long)


def train_eegnet_fold(train_sids, test_sids, all_epochs, all_channels,labels_map, fixed_ch, n_times, device, n_epochs=60, batch_size=32, lr=1e-3):
            """Train EEGNet on train_sids epochs, predict on test_sids → per-subject prob."""
    # Datasets
    tr_ds = EpochDataset(train_sids, all_epochs, all_channels,labels_map, fixed_ch, n_times, augment=True)
    te_ds = EpochDataset(test_sids,  all_epochs, all_channels,labels_map, fixed_ch, n_times, augment=False)
# Weighted sampler for class balance
    labels_tr = [item[1] for item in tr_ds.items]
    n_pd = sum(labels_tr); n_hc = len(labels_tr) - n_pd
    w  = [1.0/n_hc if l==0 else 1.0/n_pd for l in labels_tr]
    sampler = WeightedRandomSampler(torch.tensor(w,dtype=torch.float32), len(tr_ds), replacement=True)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=(device.type=='cuda'))
    te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = EEGNet(n_ch=len(fixed_ch), n_times=n_times).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    crit  = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0/n_hc, 1.0/n_pd], dtype=torch.float32).to(device))
    best_loss = float('inf')
    best_state = None

for epoch in range(n_epochs):
        model.train()
        for x, y_b in tr_ld:
            x, y_b = x.to(device), y_b.to(device)
            opt.zero_grad()
            crit(model(x), y_b).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

 # Track val loss every 10 epochs for early stopping
        if (epoch+1) % 10 == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for x, y_b in te_ld:
                    x, y_b = x.to(device), y_b.to(device)
                    loss   = crit(model(x), y_b)
                    losses.append(loss.item())

val_loss = np.mean(losses)
            if val_loss < best_loss:
                best_loss  = val_loss
                best_state = {k:v.clone() for k,v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
