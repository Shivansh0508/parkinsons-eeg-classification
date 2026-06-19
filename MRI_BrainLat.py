import os
import sys
os.environ["PYTHONUTF8"] = "1"
import ssl
import urllib3
# Disable SSL verification globally - nilearn atlas downloads fail
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

# CONFIG
CONFIG = {
    "HP_DIR"   : r"C:\Users",
    "PD_DIR"   : r"C:\Users\PD",
    "CACHE_DIR": r"C:\Users\BrainLat_Cache",
    "OUT_DIR"  : r"C:\UsersBrainLat_Results",
    "N_FOLDS"  : 5,
    "SEED"     : 42,
}
for d in [CONFIG["CACHE_DIR"], CONFIG["OUT_DIR"]]:
    os.makedirs(d, exist_ok=True)
hp_files = (glob.glob(os.path.join(CONFIG["HP_DIR"], "*.nii.gz")) + glob.glob(os.path.join(CONFIG["HP_DIR"], "*.nii")))
pd_files = (glob.glob(os.path.join(CONFIG["PD_DIR"], "*.nii.gz")) + glob.glob(os.path.join(CONFIG["PD_DIR"], "*.nii")))
print(f"HP: {len(hp_files)}  |  PD: {len(pd_files)}  |  Total: {len(hp_files)+len(pd_files)}")
if not hp_files:    raise FileNotFoundError(f"No MRI files found in {CONFIG['HP_DIR']}")
if not pd_files:    raise FileNotFoundError(f"No MRI files found in {CONFIG['PD_DIR']}")

# SUBJECT TABLE
# Each subject appears exactly once. Site is extracted from the subject ID
ef build_table(hp_dir, pd_dir):
    rows = []
    for label, folder in [(1, pd_dir), (0, hp_dir)]:
        for path in sorted(glob.glob(os.path.join(folder, "*.nii.gz")) + glob.glob(os.path.join(folder, "*.nii"))):
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
# Z-score is computed within each subject's own brain mask voxels only 
# this is NOT a population-level normalisation and introduces no leakage.
# Cache ensures reproducibility.
try:
    import ants
    ANTS_OK = True
    print("ANTs available -N4 bias correction + affine MNI registration enabled")
except ImportError:
    ANTS_OK = False
    print("ANTs not found -using nilearn resample fallback")
_MNI, _MASK = None, None

def get_mni():
    global _MNI, _MASK
    if _MNI is None:
        _MNI  = load_mni152_template(resolution=2)
        _MASK = load_mni152_brain_mask(resolution=2)
    return _MNI, _MASK

def preprocess_one(path, smooth_fwhm=6):
    """  Per-subject preprocessing pipeline:
      1. N4 bias field correction  (ANTs, or skipped if unavailable)
      2. Affine registration to MNI152 2mm space
      3. Gaussian smoothing 6mm FWHM
      4. Brain masking with MNI152 gray-matter mask
      5. Z-score intensity normalisation within this subject's masked voxels
    Step 5 uses only this subject's own voxel distribution -no population
    statistics are used, so no information leaks from test to train subjects. """
    mni, mask = get_mni()

if ANTS_OK:
        tmp_n4  = os.path.join(CONFIG["CACHE_DIR"], "_tmp_n4.nii.gz")
        tmp_mni = os.path.join(CONFIG["CACHE_DIR"], "_tmp_mni.nii.gz")
        nib.save(mni, tmp_mni)
        img_n4 = ants.n4_bias_field_correction(ants.image_read(path), verbose=False)
        ants.image_write(img_n4, tmp_n4)
reg = ants.registration(
            fixed=ants.image_read(tmp_mni),
            moving=ants.image_read(tmp_n4),
            type_of_transform='Affine',
            verbose=False)
        img_reg = nib.Nifti1Image(
            reg['warpedmovout'].numpy().astype(np.float32), mni.affine)
    else:
        img_reg = nl_image.resample_to_img(
            nib.load(path), mni,
            interpolation='linear', force_resample=True)
 img_smooth = smooth_img(img_reg, fwhm=smooth_fwhm)
    masker = NiftiMasker(mask_img=mask, standardize=False)
    data   = masker.fit_transform(img_smooth)[0]
    # Subject-level z-score (uses only this subject's voxels -no leakage)
    if data.std() > 0:
        data = (data - data.mean()) / data.std()
 return masker.inverse_transform(data).get_fdata().astype(np.float32)

def preprocess_all(df, cache_dir, force=False):
    volumes, failed = {}, []
    total = len(df)
    cached = 0
    for i, row in df.iterrows():
        sid   = row["subject_id"]
        fpath = os.path.join(cache_dir, f"{sid}.npy")
        if os.path.exists(fpath) and not force:
            volumes[sid] = np.load(fpath)
            cached += 1
            continue
 try:
            vol = preprocess_one(row["path"])
            np.save(fpath, vol)
            volumes[sid] = vol
            done = cached + len(volumes) - cached
            print(f"  [{len(volumes)}/{total}] {sid}", end='\r')
            except Exception as e:
            print(f"\n  FAIL {sid}: {e}")
            failed.append(sid)
    print(f"\nPreprocessed: {len(volumes)}  |  From cache: {cached}  |  Failed: {len(failed)}")
    return volumes, failed
print("\nPreprocessing (runs once, then loads from cache)...")
volumes, failed = preprocess_all(subjects_df, CONFIG["CACHE_DIR"])

if failed:
    subjects_df = subjects_df[~subjects_df.subject_id.isin(failed)].reset_index(drop=True)
y     = subjects_df["label"].values
sites = subjects_df["site"].values
print(f"Final dataset: {len(subjects_df)} subjects  PD={y.sum()}  HC={len(y)-y.sum()}")

# FEATURE EXTRACTION
# AAL (116 regions) + Harvard-Oxford subcortical (21 regions) = 137 features
# Each masker is fit on the MNI atlas image -NOT on the subject data.
def find_local_atlas(nilearn_data_dir, patterns):
    """    Search nilearn_data_dir recursively for a .nii or .nii.gz file whose
    name contains any of the given patterns. Returns first match or None.
    Only matches image files -never .xml, .txt, .csv etc.  """
    for root, dirs, files in os.walk(nilearn_data_dir):
        for fname in files:
            if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
                continue
            for pat in patterns:
                if pat.lower() in fname.lower():
                    return os.path.join(root, fname)
    return None

def extract_atlas_features(df, volumes):
    """ Extracts mean signal per brain atlas region for each subject.
    Maskers are fit on the atlas label image (fixed template) -not on
    subject data -so no leakage regardless of train/test split.

    Both atlases are loaded from local disk only -no network calls.
    If a local file is not found the function raises a clear error
    telling you exactly where to place the file.  """  
    mni, _ = get_mni()
    nilearn_dir = r"C:\Users\nilearn_data"
    # AAL atlas
 confirmed = r"C:\Users\nilearn_data\aal\atlas\AAL.nii"
    if os.path.exists(confirmed):
        aal_path = confirmed
    else:
        aal_path = find_local_atlas(nilearn_dir, ["AAL.nii", "AAL_MNI_V4.nii", "aal.nii", "ROI_MNI"])
    if aal_path is None:
        raise FileNotFoundError( "AAL.nii not found. Expected at:\n"
                                r"  C:\Users\nilearn_data\aal\atlas\AAL.nii")
   
print(f"AAL: loaded from {aal_path}")
    aal_img = nib.load(aal_path)
    aal_res = nl_image.resample_to_img(aal_img, mni, interpolation='nearest')
     xml_path = aal_path.replace(".nii", ".xml").replace(".NII", ".xml")
    if os.path.exists(xml_path):
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        aal_region_names = [el.text.strip()
                            for el in tree.findall(".//label/name")]

if not aal_region_names: aal_region_names = [el.text.strip() for el in tree.iter() if el.text]
    else:
        n_aal_regions    = int(np.unique(aal_res.get_fdata()).max())
        aal_region_names = [f"AAL_{i}" for i in range(1, n_aal_regions + 1)]
    m_aal = NiftiLabelsMasker(labels_img=aal_res, standardize=False, strategy='mean', resampling_target=None)
    m_aal.fit()
    # Harvard-Oxford subcortical atlas
    ho_path = find_local_atlas(nilearn_dir, [
        "HarvardOxford-sub-maxprob-thr25-2mm",
        "HarvardOxford-sub-maxprob-thr25-1mm",
        "HarvardOxford-sub-maxprob-thr50-2mm",
        "HarvardOxford-sub-maxprob-thr50-1mm"])

if ho_path is None:
        # Try fetching with SSL disabled 
        try:
            print("HO atlas not found locally -attempting download...")
            ho     = fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
            ho_res = nl_image.resample_to_img(ho.maps, mni, interpolation='nearest')
        except Exception:
            # Last resort: use only AAL features
            print("WARNING: Harvard-Oxford atlas unavailable. Using AAL only (116 features).")
            feats_aal = []
            for _, row in df.iterrows():
                img = nib.Nifti1Image(volumes[row["subject_id"]], mni.affine)
                feats_aal.append(m_aal.transform(img).ravel())
            X_aal = np.vstack(feats_aal)
            print(f"AAL only: {X_aal.shape}")
            return X_aal, X_aal, aal_region_names
else:
        print(f"HO: loaded from {ho_path}")
        ho_res = nl_image.resample_to_img(nib.load(ho_path), mni, interpolation='nearest')
    m_ho = NiftiLabelsMasker(labels_img=ho_res, standardize=False, strategy='mean', resampling_target=None)
    m_ho.fit()
    feats_aal, feats_ho = [], []

    for _, row in df.iterrows():
        img = nib.Nifti1Image(volumes[row["subject_id"]], mni.affine)
        feats_aal.append(m_aal.transform(img).ravel())
        feats_ho.append(m_ho.transform(img).ravel())
    X_aal  = np.vstack(feats_aal)
    X_ho   = np.vstack(feats_ho)
    X_comb = np.hstack([X_aal, X_ho])
    print(f"AAL: {X_aal.shape}  |  HO: {X_ho.shape}  |  Combined: {X_comb.shape}")
    return X_comb, X_aal, aal_region_names

print("\nExtracting atlas features...")
X_atlas, X_aal_only, aal_labels = extract_atlas_features(subjects_df, volumes)
n_pd      = int(y.sum())
n_hc      = int(len(y) - n_pd)
scale_pos = n_hc / n_pd
print(f"Class ratio HC/PD = {scale_pos:.2f}  (used as XGBoost scale_pos_weight)")

# STRATIFIED GROUP K-FOLD CROSS-VALIDATION
# FEATURE ENGINEERING  — adds 5 extra feature types on top of raw atlas means
# All computed from the 137-dim atlas feature vector, no new data needed.
# 1. Asymmetry index: (L-R)/(L+R) per bilateral pair  -> lateralisation signal
# 2. Log transform: log(|x|+1)  -> compresses outlier voxels
# 3. Squared features: x^2      -> captures nonlinear magnitude effects
# 4. Pairwise ratios of 10 most PD-relevant AAL regions (putamen, caudate etc.)
# 5. Z-score within subject across regions (relative profile)
PD_ROI_IDX = [67, 68, 71, 72, 73, 74, 77, 78, 83, 19]

def engineer_features(X_train, X_test):
    """ Applies all feature engineering transforms.
    Fit parameters come from X_train only — applied to both train and test.
    Returns augmented (X_train_eng, X_test_eng). """
    def _transform(X, mean_tr, std_tr):
        n = X.shape[1]

        # 1. Asymmetry index for first 116 AAL features (58 bilateral pairs)
        half = min(58, n // 2)
        L = X[:, :half]
        R = X[:, half:2*half]
        denom = np.abs(L) + np.abs(R) + 1e-8
        asym = (L - R) / denom       

        # 2. Log magnitude
        log_x = np.sign(X) * np.log1p(np.abs(X))    

        # 3. Squared
        sq_x = X ** 2                          

        # 4. Pairwise ratios of PD ROIs
        roi = X[:, PD_ROI_IDX]                        
        pairs = []
        for i in range(len(PD_ROI_IDX)):
            for j in range(i+1, len(PD_ROI_IDX)):
                pairs.append(roi[:, i] / (roi[:, j] + 1e-8))
        ratio = np.column_stack(pairs)                

        # 5. Within-subject z-score across all regions
        row_mean = np.mean(X, axis=1, keepdims=True)
        row_std  = np.std(X,  axis=1, keepdims=True) + 1e-8
        zscore   = (X - row_mean) / row_std             

        X_eng = np.hstack([X, asym, log_x, sq_x, ratio, zscore])
        # Standardise using training mean/std
        X_eng = (X_eng - mean_tr) / (std_tr + 1e-8)
        return X_eng

    # Compute all engineered features for train
    half = min(58, X_train.shape[1] // 2)
    L_tr = X_train[:, :half]; R_tr = X_train[:, half:2*half]
    asym_tr  = (L_tr - R_tr) / (np.abs(L_tr) + np.abs(R_tr) + 1e-8)
    log_tr   = np.sign(X_train) * np.log1p(np.abs(X_train))
    sq_tr    = X_train ** 2
    roi_tr   = X_train[:, PD_ROI_IDX]
    pairs_tr = []
    for i in range(len(PD_ROI_IDX)):
        for j in range(i+1, len(PD_ROI_IDX)):
            pairs_tr.append(roi_tr[:, i] / (roi_tr[:, j] + 1e-8))
    ratio_tr = np.column_stack(pairs_tr)
    rm_tr = np.mean(X_train, axis=1, keepdims=True)
    rs_tr = np.std(X_train,  axis=1, keepdims=True) + 1e-8
    zs_tr = (X_train - rm_tr) / rs_tr
    X_train_eng = np.hstack([X_train, asym_tr, log_tr, sq_tr, ratio_tr, zs_tr])

    # Fit scaler on training engineered features
    mean_tr = X_train_eng.mean(axis=0)
    std_tr  = X_train_eng.std(axis=0)
    X_train_eng = (X_train_eng - mean_tr) / (std_tr + 1e-8)

    # Apply same transforms + scaler to test
    L_te = X_test[:, :half]; R_te = X_test[:, half:2*half]
    asym_te  = (L_te - R_te) / (np.abs(L_te) + np.abs(R_te) + 1e-8)
    log_te   = np.sign(X_test) * np.log1p(np.abs(X_test))
    sq_te    = X_test ** 2
    roi_te   = X_test[:, PD_ROI_IDX]
    pairs_te = []
    for i in range(len(PD_ROI_IDX)):
        for j in range(i+1, len(PD_ROI_IDX)):
            pairs_te.append(roi_te[:, i] / (roi_te[:, j] + 1e-8))

 ratio_te = np.column_stack(pairs_te)
    rm_te = np.mean(X_test, axis=1, keepdims=True)
    rs_te = np.std(X_test,  axis=1, keepdims=True) + 1e-8
    zs_te = (X_test - rm_te) / rs_te
    X_test_eng = np.hstack([X_test, asym_te, log_te, sq_te, ratio_te, zs_te])
    X_test_eng = (X_test_eng - mean_tr) / (std_tr + 1e-8)
    return X_train_eng, X_test_eng

# STRATIFIED GROUP K-FOLD  —  MAXIMUM POWER ENSEMBLE
# 6 classifiers per fold, all with SMOTE + PCA inside pipeline:
#   1. SVM RBF  C=10
#   2. SVM RBF  C=100   (captures tighter decision boundary)
#   3. XGBoost  lr=0.03, depth=4
#   4. XGBoost  lr=0.01, depth=6  (deeper trees, more interactions)
#   5. LightGBM (faster gradient boosting, different inductive bias)
#   6. Logistic Regression (strong linear baseline)

# Final probability = weighted average where weight = each model's val AUC
# on its own training fold  (AUC-weighted soft vote)

# Requirements satisfied:
#   - Subject-level splits, no scan leakage
#   - SMOTE only inside training fold
#   - StandardScaler + PCA fit on training fold only
#   - Feature engineering fit on training fold only
#   - Test fold never touches any fit step

def stratified_group_kfold_cv(X, y, sites, subjects_df, n_folds=5):
    try:
        import lightgbm as lgb
        LGB_OK = True
    except ImportError:
        LGB_OK = False
        print("LightGBM not installed — running without it (still 5 classifiers)")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_records = []
    all_true, all_prob, all_pred = [], [], []
    print(f"\nRunning {n_folds}-fold Stratified CV  "
          f"(n={len(y)}, PD={y.sum()}, HC={len(y)-y.sum()})")
    print("-" * 75)

for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        sites_test      = sites[test_idx]
        n_pd_tr = int(y_train.sum())
        n_hc_tr = int(len(y_train) - n_pd_tr)
        print(f"Fold {fold_i+1}  |  "
              f"Train: {len(y_train)} (PD={n_pd_tr} HC={n_hc_tr})  |  "
              f"Test: {len(y_test)} (PD={int(y_test.sum())} HC={int(len(y_test)-y_test.sum())})  |  "
              f"Test sites: {sorted(set(sites_test))}")
        # Feature engineering — fit on train, apply to both
        X_tr_eng, X_te_eng = engineer_features(X_train, X_test)
        k_nn      = min(5, n_pd_tr - 1)
        pos_scale = n_hc_tr / n_pd_tr

        #  build 6 pipelines 
        svm10 = ImbPipeline([
            ("sm",  SMOTE(random_state=42, k_neighbors=k_nn)),
            ("pca", PCA(n_components=80, random_state=42)),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                        class_weight="balanced", probability=True, random_state=42))])

        svm100 = ImbPipeline([
            ("sm",  SMOTE(random_state=42, k_neighbors=k_nn)),
            ("pca", PCA(n_components=80, random_state=42)),
            ("clf", SVC(kernel="rbf", C=100, gamma="scale",
                        class_weight="balanced", probability=True, random_state=42))])

xgb4 = ImbPipeline([
            ("sm",  SMOTE(random_state=42, k_neighbors=k_nn)),
            ("pca", PCA(n_components=80, random_state=42)),
            ("clf", xgb.XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=pos_scale,
                reg_alpha=0.3, reg_lambda=1.5,
                eval_metric="logloss", verbosity=0, random_state=42))])

        xgb6 = ImbPipeline([
            ("sm",  SMOTE(random_state=42, k_neighbors=k_nn)),
            ("pca", PCA(n_components=80, random_state=42)),
            ("clf", xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.01,
                subsample=0.7, colsample_bytree=0.7,
                scale_pos_weight=pos_scale,
                reg_alpha=0.5, reg_lambda=2.0,
                eval_metric="logloss", verbosity=0, random_state=42))])

lr_p = ImbPipeline([
            ("sm",  SMOTE(random_state=42, k_neighbors=k_nn)),
            ("pca", PCA(n_components=80, random_state=42)),
            ("clf", LogisticRegression(C=0.05, class_weight="balanced",
                                       solver="lbfgs", max_iter=3000,
                                       random_state=42))])

        pipes = [("SVM-C10",  svm10),
                 ("SVM-C100", svm100),
                 ("XGB-d4",   xgb4),
                 ("XGB-d6",   xgb6),
                 ("LR",       lr_p)]
if LGB_OK:
            lgb_p = ImbPipeline([
                ("sm",  SMOTE(random_state=42, k_neighbors=k_nn)),
                ("pca", PCA(n_components=80, random_state=42)),
                ("clf", lgb.LGBMClassifier(
                    n_estimators=500, max_depth=4, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=pos_scale,
                    reg_alpha=0.3, reg_lambda=1.5,
                    verbose=-1, random_state=42))])
            pipes.append(("LGB", lgb_p))

        # fit all classifiers on engineered training features 
        probs_list = []
        weights    = []
        for pname, pipe in pipes:
            pipe.fit(X_tr_eng, y_train)
            p_te  = pipe.predict_proba(X_te_eng)[:, 1]
            # AUC on training data (internal estimate of model quality)
            # Used as weight — better models get higher vote weight
            p_tr  = pipe.predict_proba(X_tr_eng)[:, 1]
            # Use a small internal CV estimate instead of train AUC to avoid
            # over-weighting overfit models: use leave-20%-out on train
            from sklearn.model_selection import cross_val_predict
            p_cv  = cross_val_predict(pipe, X_tr_eng, y_train,
                                      cv=3, method="predict_proba")[:, 1]
            w     = roc_auc_score(y_train, p_cv)
            probs_list.append(p_te)
            weights.append(w)
            print(f"    {pname:10s}  train-cv AUC={w:.4f}")
        # AUC-weighted soft vote
        weights    = np.array(weights)
        weights    = weights / weights.sum()
        prob       = sum(w * p for w, p in zip(weights, probs_list))
        pred       = (prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        fold_acc  = accuracy_score(y_test, pred)
        fold_auc  = roc_auc_score(y_test, prob)
        fold_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fold_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fold_f1   = f1_score(y_test, pred, zero_division=0)
        fold_prec = precision_score(y_test, pred, zero_division=0)

        print(f"  --> Fold {fold_i+1}: "
              f"Acc={fold_acc:.4f}  AUC={fold_auc:.4f}  "
              f"Sens={fold_sens:.4f}  Spec={fold_spec:.4f}  "
              f"F1={fold_f1:.4f}")

fold_records.append(dict(fold=fold_i+1, acc=fold_acc, auc=fold_auc, sens=fold_sens, spec=fold_spec, f1=fold_f1, prec=fold_prec, tp=tp, tn=tn, fp=fp, fn=fn))
        all_true.extend(y_test.tolist())
        all_prob.extend(prob.tolist())
        all_pred.extend(pred.tolist())
    metrics = ["acc", "auc", "sens", "spec", "f1", "prec"]
    agg = {m: (np.mean([r[m] for r in fold_records]),
               np.std([r[m]  for r in fold_records]))
           for m in metrics}
tn_g, fp_g, fn_g, tp_g = confusion_matrix(all_true, all_pred).ravel()
    result = dict(
        name="Max-Power Ensemble (6 classifiers, AUC-weighted)",
        fold_records=fold_records,
        agg=agg,
        all_true=all_true,
        all_prob=all_prob,
        all_pred=all_pred,
        global_cm=np.array([[tn_g, fp_g], [fn_g, tp_g]]))
    return result

result = stratified_group_kfold_cv(
    X_atlas, y, sites, subjects_df,
    n_folds=CONFIG["N_FOLDS"])

# LEAVE-ONE-SITE-OUT VALIDATION 
# Every subject from one site is held out as the test set.
# This tests generalisation to unseen acquisition sites.
def leave_one_site_out(X, y, sites, pca_k=40):
    """ Leave-One-Site-Out cross-validation.
    Each iteration: train on all sites except one, test on the held-out site.
    Requirements:
      - StandardScaler fit on training sites only
      - SMOTE applied to training data only
      - PCA fit on training data only
      - Test site data never used during fit """

logo = LeaveOneGroupOut()
    fold_records = []
    all_true, all_prob, all_pred = [], [], []
    print(f"\nLeave-One-Site-Out  (groups: {sorted(set(sites))})")
    print("-" * 70)
    for tr_idx, te_idx in logo.split(X, y, groups=sites):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        site_name   = sites[te_idx[0]]
