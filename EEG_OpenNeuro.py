# EEG PD vs HC  —  DS007526  —  Maximum Accuracy Version
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
