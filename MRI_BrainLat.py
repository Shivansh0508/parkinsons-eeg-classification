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
