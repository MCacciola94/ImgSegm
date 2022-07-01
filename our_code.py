import os 
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff

import warnings
warnings.simplefilter("ignore")

from paths_config import config_org as config
from utils.utils import *

data = pd.read_csv(config.TRAIN_CSV_PATH)

df_prostate = data.loc[data['organ']=='prostate']
df_spleen = data.loc[data['organ']=='spleen']
df_lung = data.loc[data['organ']=='lung']
df_kidney = data.loc[data['organ']=='kidney']
df_LI = data.loc[data['organ']=='largeintestine']


