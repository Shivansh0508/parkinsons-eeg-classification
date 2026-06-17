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
    """
    Per-subject preprocessing pipeline:
      1. N4 bias field correction  (ANTs, or skipped if unavailable)
      2. Affine registration to MNI152 2mm space
      3. Gaussian smoothing 6mm FWHM
      4. Brain masking with MNI152 gray-matter mask
      5. Z-score intensity normalisation within this subject's masked voxels

    Step 5 uses only this subject's own voxel distribution -no population
    statistics are used, so no information leaks from test to train subjects.
    """
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
    """
    Search nilearn_data_dir recursively for a .nii or .nii.gz file whose
    name contains any of the given patterns. Returns first match or None.
    Only matches image files -never .xml, .txt, .csv etc.
    """
    for root, dirs, files in os.walk(nilearn_data_dir):
        for fname in files:
            if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
                continue
            for pat in patterns:
                if pat.lower() in fname.lower():
                    return os.path.join(root, fname)
    return None

def extract_atlas_features(df, volumes):
    """
    Extracts mean signal per brain atlas region for each subject.
    Maskers are fit on the atlas label image (fixed template) -not on
    subject data -so no leakage regardless of train/test split.

    Both atlases are loaded from local disk only -no network calls.
    If a local file is not found the function raises a clear error
    telling you exactly where to place the file.
    """
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

    m_aal = NiftiLabelsMasker(labels_img=aal_res, standardize=False,
                              strategy='mean', resampling_target=None)
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
