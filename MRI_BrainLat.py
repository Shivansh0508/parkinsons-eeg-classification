import os
import sys
os.environ["PYTHONUTF8"] = "1"
import ssl
import urllib3

# Disable SSL verification globally — nilearn atlas downloads fail
# on Windows with corporate/institutional certificates
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch requests (used internally by nilearn) to skip SSL verify
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
from nilearn.datasets import (load_mni152_template, load_mni152_brain_mask,
                               fetch_atlas_aal, fetch_atlas_harvard_oxford)
