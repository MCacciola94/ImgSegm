# %% [markdown] {"papermill":{"duration":0.01772,"end_time":"2021-03-31T10:37:28.865319","exception":false,"start_time":"2021-03-31T10:37:28.847599","status":"completed"},"tags":[]}
#   --- updated data, old models ---  
# * v01(old v53) : 10_01 with fold0,1,2,3, LB=(old 0.874)(th=0.5, pad=256, tta2) -> Timeout Error  
# * v02(old v53) : rasterIO, 10_01 with fold0, LB=(old ?)(th=0.5, pad=256, tta2) -> Exceeded Error  
# * v03(old v53) : rasterIO, 10_01 with fold0,1,2,3, LB=(old 0.874)(th=0.5, pad=256, tta2)  
# * v04 : rasterIO, 10_01 with fold0, LB= (th=0.5, pad=64, no tta)  
# * v05 : debug  
# * v06 : re-upload the weights, rasterIO, 10_01 with fold0, LB= (th=0.5, pad=64, no tta)  
# * v09 : debug, sample submission -> OK  
# * v10 : rasterIO, 10_01 with fold0, LB= (th=0.5, pad=0, no tta)  
# * v11 : bug-corrected, rasterIO, 10_01 with fold0, LB= (th=0.5, pad=0, no tta)  
#   
#   
#   --- OK ---  
# * v14 : better approach, rasterIO, 10_01 with fold0, LB=0.908 (th=0.5, input_sz=256, pad=0, no tta)  
# * v18 : better approach, rasterIO, 10_01 with fold0123, LB= (th=0.5, input_sz=320, pad=0, tta2) <-- GPU OOM  
# * v17 : better approach, rasterIO, 10_01 with fold0123456, LB=0.920 (th=0.5, input_sz=320, pad=0, tta4)  
# * v21 : better approach, rasterIO, 10_01 with fold0123, LB=0.924 (th=0.5, input_sz=320, pad=64, tta2)  
# * v22 : better approach, rasterIO, 10_01 with fold0123, LB=0.923 (th=0.5, input_sz=320, pad=128, tta2)  
# * v23 : better approach, rasterIO, 10_01 with fold0123, LB=0.924 (th=0.5, input_sz=320, pad=256, tta2)  
# * v24 : better approach, rasterIO, 10_01 with fold0123456, LB=0.923 (th=0.5, input_sz=320, pad=256, tta2)  
# * v25 : better approach, rasterIO, 10_01 with fold0123, LB= (th=0.5, input_sz=320, pad=256, tta4) -> TLO  
# * v26 : better approach, rasterIO, 10_01 with fold0123, LB=0.925 (th=0.5, input_sz=320, pad=256, tta3)  
#   
#   
#   --- trained with new data ---  
# * v28 : new_01_01 : baseline with new kaggle dataset, unet seresnext101, more hue-sat(0.3), classification head, deepsupervision, 320x320, coarse dropout corrected, 1024x1024 tiling, 5folds, data balance with (00_01, 00_02), group by patient number, balanced sampling, aug(), cosine_annealing(1e-4to1e-6, 20epochs), CV=0.9354, LB=0.917(th=0.5, input_sz=320, pad=256, tta2)  
# * v29 : new_01_01 : baseline with new kaggle dataset, unet seresnext101, more hue-sat(0.3), classification head, deepsupervision, 320x320, coarse dropout corrected, 1024x1024 tiling, 5folds, data balance with (00_01, 00_02), group by patient number, balanced sampling, aug(), cosine_annealing(1e-4to1e-6, 20epochs), CV=0.9354, LB=0.921(th=0.5, input_sz=320, pad=64, tta4)  
# * v30 : 01_02 : manually specified val_patient_numbers (4folds), baseline with new kaggle dataset, unet seresnext101, more hue-sat(0.3), classification head, deepsupervision, 320x320, coarse dropout corrected, 1024x1024 tiling, data balance with (00_02, 00_04), group by patient number, balanced sampling, aug(), cosine_annealing(1e-4to1e-6, 20epochs), CV=0.9331, LB=0.921(th=0.5, input_sz=320, pad=256, tta3)  
# * v31 : 01_04 : unet seresnext50, manually specified val_patient_numbers (4folds), baseline with new kaggle dataset, more hue-sat(0.3), classification head, deepsupervision, 320x320, coarse dropout corrected, 1024x1024 tiling, data balance with (00_02, 00_04), group by patient number, balanced sampling, aug(), cosine_annealing(1e-4to1e-6, 20epochs), CV=0.9327, LB=0.921(th=0.5, input_sz=320, pad=256, tta3)  
# * v33 : 03_01 : pseudo-label for train+test(00_05,00_06), manually specified val_patient_numbers (4folds), baseline with new kaggle dataset, unet seresnext101, more hue-sat(0.3), classification head, deepsupervision, 320x320, coarse dropout corrected, 1024x1024 tiling, data balance, group by patient number, balanced sampling, aug(), cosine_annealing(1e-4to1e-6, 20epochs), CV=0.9779, LB=0.921(th=0.5, input_sz=320, pad=256, tta3)  
# * v35 : v30(LB0.921), create sub for LB probing  
# * v36 : v26(LB0.925), create sub for LB probing  
# * v37 : v33(LB0.921), create sub for LB probing  
# * v38 : 01_07 : focal loss, manually specified val_patient_numbers (4folds), baseline with new kaggle dataset, unet seresnext101, more hue-sat(0.3), classification head, deepsupervision, 320x320, coarse dropout corrected, 1024x1024 tiling, data balance with (00_02, 00_04), group by patient number, balanced sampling, aug(), cosine_annealing(1e-4to1e-6, 20epochs), CV=0.9326, LB=0.921(th=0.5, input_sz=320, pad=256, tta3)  
# * v39 : v38(LB=), create sub for LB probing  
# * v41 : 03_02 : pseudo-label (test(00_05,00_06), external(00,10,00_11), dataset_a_dib(00_08,00_09)), manually specified val_patient_numbers (4folds), baseline with new kaggle dataset, unet seresnext101, more hue-sat(0.3), classification head, deepsupervision, 320x320, coarse dropout corrected, 1024x1024 tiling, data balance, group by patient number, balanced sampling, aug(), cosine_annealing(1e-4to1e-6, 20epochs), CV=0.9385  (fold2 looks not good, should be removed ?), LB=0.922(th=0.5, input_sz=320, pad=256, tta3)  
# * v42 : fold013 of 03_02 : pseudo-label (test(00_05,00_06), external(00,10,00_11), dataset_a_dib(00_08,00_09)), manually specified val_patient_numbers (4folds), baseline with new kaggle dataset, unet seresnext101, more hue-sat(0.3), classification head, deepsupervision, 320x320, coarse dropout corrected, 1024x1024 tiling, data balance, group by patient number, balanced sampling, aug(), cosine_annealing(1e-4to1e-6, 20epochs), CV=0.9385  (fold2 looks not good, should be removed ?), LB=0.922(th=0.5, input_sz=320, pad=256, tta4)  
# * v43 : 03_03 : add Carno Chao's hand-labeling data (d488c759a), pseudo-label (test(00_05,00_06), external(00,10,00_11), dataset_a_dib(00_08,00_09)), manually specified val_patient_numbers (4folds), baseline with new kaggle dataset, unet seresnext101, more hue-sat(0.3), classification head, deepsupervision, 320x320, coarse dropout corrected, 1024x1024 tiling, data balance, group by patient number, balanced sampling, aug(), cosine_annealing(1e-4to1e-6, 20epochs), CV=0.9375* v41 : 03_02 : pseudo-label (test(00_05,00_06), external(00,10,00_11), dataset_a_dib(00_08,00_09)), manually specified val_patient_numbers (4folds), baseline with new kaggle dataset, unet seresnext101, more hue-sat(0.3), classification head, deepsupervision, 320x320, coarse dropout corrected, 1024x1024 tiling, data balance, group by patient number, balanced sampling, aug(), cosine_annealing(1e-4to1e-6, 20epochs), CV=0.9385  (fold2 looks not good, should be removed ?), LB=(th=0.5, input_sz=320, pad=256, tta3)  

# %% [markdown] {"papermill":{"duration":0.015995,"end_time":"2021-03-31T10:37:28.897826","exception":false,"start_time":"2021-03-31T10:37:28.881831","status":"completed"},"tags":[]}
# # Config

# %% [code] {"papermill":{"duration":0.024862,"end_time":"2021-03-31T10:37:28.939348","exception":false,"start_time":"2021-03-31T10:37:28.914486","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:44:52.646319Z","iopub.execute_input":"2022-06-28T19:44:52.646612Z","iopub.status.idle":"2022-06-28T19:44:52.651613Z","shell.execute_reply.started":"2022-06-28T19:44:52.646584Z","shell.execute_reply":"2022-06-28T19:44:52.650127Z"}}
DEBUG = False

# %% [code] {"papermill":{"duration":1.689504,"end_time":"2021-03-31T10:37:30.64523","exception":false,"start_time":"2021-03-31T10:37:28.955726","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:44:54.461551Z","iopub.execute_input":"2022-06-28T19:44:54.461876Z","iopub.status.idle":"2022-06-28T19:44:55.907435Z","shell.execute_reply.started":"2022-06-28T19:44:54.461845Z","shell.execute_reply":"2022-06-28T19:44:55.906673Z"}}
import torch
from paths_config import config_kid 

config = {
    'split_seed_list':[0],
    'FOLD_LIST':[0,1,2,3], 
    'model_path': config_kid.BASE_PATH / 'hubmap-new-03-03/',
    'model_name':'seresnext101', #resnet34
    
    'num_classes':1,
    'resolution':1024, #(1024,1024),(512,512),
    'input_resolution':320, #(320,320), #(256,256), #(512,512), #(384,384)
    'deepsupervision':False, # always false for inference
    'clfhead':False,
    'clf_threshold':0.5,
    'small_mask_threshold':0, #256*256*0.03, #512*512*0.03,
    'mask_threshold':0.5,
    'pad_size':256, #(64,64), #(256,256), #(128,128)
    
    'tta':3,
    'test_batch_size':12,
    
    'FP16':False,
    'num_workers':4,
    'device':"cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

device = config['device']

# %% [markdown] {"papermill":{"duration":0.016215,"end_time":"2021-03-31T10:37:30.677231","exception":false,"start_time":"2021-03-31T10:37:30.661016","status":"completed"},"tags":[]}
# # Import Libraries and Data

# %% [code] {"papermill":{"duration":0.459661,"end_time":"2021-03-31T10:37:31.152738","exception":false,"start_time":"2021-03-31T10:37:30.693077","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:45:00.676041Z","iopub.execute_input":"2022-06-28T19:45:00.678597Z","iopub.status.idle":"2022-06-28T19:45:01.237604Z","shell.execute_reply.started":"2022-06-28T19:45:00.678552Z","shell.execute_reply":"2022-06-28T19:45:01.236696Z"}}
import numpy as np
import pandas as pd
pd.get_option("display.max_columns")
pd.set_option('display.max_columns', 300)
pd.get_option("display.max_rows")
pd.set_option('display.max_rows', 300)

import matplotlib.pyplot as plt
#%matplotlib inline

import sys
import os
from os.path import join as opj
import gc

import cv2
import rasterio
from rasterio.windows import Window

INPUT_PATH = config_kid.DSET_PATH

# %% [code] {"papermill":{"duration":0.028018,"end_time":"2021-03-31T10:37:31.199056","exception":false,"start_time":"2021-03-31T10:37:31.171038","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:45:05.704901Z","iopub.execute_input":"2022-06-28T19:45:05.705245Z","iopub.status.idle":"2022-06-28T19:45:05.712208Z","shell.execute_reply.started":"2022-06-28T19:45:05.705211Z","shell.execute_reply":"2022-06-28T19:45:05.711365Z"}}
print('Python        : ' + sys.version.split('\n')[0])
print('Numpy         : ' + np.__version__)
print('Pandas        : ' + pd.__version__)
print('Rasterio      : ' + rasterio.__version__)
print('OpenCV        : ' + cv2.__version__)

# %% [code] {"papermill":{"duration":0.369479,"end_time":"2021-03-31T10:37:31.58586","exception":false,"start_time":"2021-03-31T10:37:31.216381","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:45:07.621053Z","iopub.execute_input":"2022-06-28T19:45:07.621416Z","iopub.status.idle":"2022-06-28T19:45:07.943629Z","shell.execute_reply.started":"2022-06-28T19:45:07.621382Z","shell.execute_reply":"2022-06-28T19:45:07.942888Z"}}
train_df = pd.read_csv(config_kid.TRAIN_CSV_PATH)
info_df  = pd.read_csv(INPUT_PATH /'HuBMAP-20-dataset_information.csv')
sub_df = pd.read_csv(INPUT_PATH / 'sample_submission.csv')

print('train_df.shape = ', train_df.shape)
print('info_df.shape  = ', info_df.shape)
print('sub_df.shape = ', sub_df.shape)

# %% [code] {"papermill":{"duration":0.025261,"end_time":"2021-03-31T10:37:31.62786","exception":false,"start_time":"2021-03-31T10:37:31.602599","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:45:09.860502Z","iopub.execute_input":"2022-06-28T19:45:09.860819Z","iopub.status.idle":"2022-06-28T19:45:09.866075Z","shell.execute_reply.started":"2022-06-28T19:45:09.860789Z","shell.execute_reply":"2022-06-28T19:45:09.865156Z"}}
#sub_df['predicted'] = '1 1'
#sub_df.to_csv('submission.csv', index=False)

if len(sub_df) == 5:
    if DEBUG:
        sub_df = sub_df[:]
    else:
        sub_df = sub_df[:1]

# %% [markdown] {"papermill":{"duration":0.016345,"end_time":"2021-03-31T10:37:31.660344","exception":false,"start_time":"2021-03-31T10:37:31.643999","status":"completed"},"tags":[]}
# # Utils  

# %% [code] {"papermill":{"duration":0.030782,"end_time":"2021-03-31T10:37:31.707493","exception":false,"start_time":"2021-03-31T10:37:31.676711","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:45:12.360078Z","iopub.execute_input":"2022-06-28T19:45:12.360556Z","iopub.status.idle":"2022-06-28T19:45:12.379047Z","shell.execute_reply.started":"2022-06-28T19:45:12.360509Z","shell.execute_reply":"2022-06-28T19:45:12.377811Z"}}
import random
import torch
import numpy as np
import os
import time

def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def elapsed_time(start_time):
    return time.time() - start_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

fix_seed(2020)

# %% [code] {"papermill":{"duration":0.033552,"end_time":"2021-03-31T10:37:31.757271","exception":false,"start_time":"2021-03-31T10:37:31.723719","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:45:14.805661Z","iopub.execute_input":"2022-06-28T19:45:14.805984Z","iopub.status.idle":"2022-06-28T19:45:14.819768Z","shell.execute_reply.started":"2022-06-28T19:45:14.805954Z","shell.execute_reply":"2022-06-28T19:45:14.818164Z"}}
import cv2

def rle2mask(rle, shape):
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return 
    Returns numpy array <- 1(mask), 0(background)
    '''
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Needed to align to RLE direction


def mask2rle(img, shape, small_mask_threshold):
    '''
    Convert mask to rle.
    img: numpy array <- 1(mask), 0(background)
    Returns run length as string formated
    
    pixels = np.array([1,1,1,0,0,1,0,1,1]) #-> rle = '1 3 6 1 8 2'
    pixels = np.concatenate([[0], pixels, [0]]) #[0,1,1,1,0,0,1,0,1,1,0]
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1 #[ 1  4  6  7  8 10] bit change points
    print(runs[1::2]) #[4 7 10]
    print(runs[::2]) #[1 6 8]
    runs[1::2] -= runs[::2]
    print(runs) #[1 3 6 1 8 2]
    '''
    if img.shape != shape:
        h,w = shape
        img = cv2.resize(img, dsize=(w,h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.int8) 
    pixels = img.T.flatten()
    #pixels = np.concatenate([[0], pixels, [0]])
    pixels = np.pad(pixels, ((1, 1), ))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if runs[1::2].sum() <= small_mask_threshold:
        return ''
    else:
        return ' '.join(str(x) for x in runs)

# %% [markdown] {"papermill":{"duration":0.01607,"end_time":"2021-03-31T10:37:31.789392","exception":false,"start_time":"2021-03-31T10:37:31.773322","status":"completed"},"tags":[]}
# # Model

# %% [code] {"papermill":{"duration":1.551183,"end_time":"2021-03-31T10:37:33.356746","exception":false,"start_time":"2021-03-31T10:37:31.805563","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:45:17.627267Z","iopub.execute_input":"2022-06-28T19:45:17.627623Z","iopub.status.idle":"2022-06-28T19:45:19.169045Z","shell.execute_reply.started":"2022-06-28T19:45:17.627587Z","shell.execute_reply":"2022-06-28T19:45:19.168248Z"}}
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
package_dir = str(config_kid.BASE_PATH / "pretrainedmodels/pretrained-models.pytorch-master/")
sys.path.insert(0, package_dir)
import pretrainedmodels


def conv3x3(in_channel, out_channel): #not change resolusion
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=3,stride=1,padding=1,dilation=1,bias=False)

def conv1x1(in_channel, out_channel): #not change resolution
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=1,stride=1,padding=0,dilation=1,bias=False)

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform_(m.weight, gain=1)
        #nn.init.xavier_normal_(m.weight, gain=1)
        #nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1,0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)

        
class cSEBlock(nn.Module):
    def __init__(self, c, feat):
        super().__init__()
        self.attention_fc = nn.Linear(feat,1, bias=False)
        self.bias         = nn.Parameter(torch.zeros((1,c,1), requires_grad=True))
        self.sigmoid      = nn.Sigmoid()
        self.dropout      = nn.Dropout2d(0.1)
        
    def forward(self,inputs):
        batch,c,h,w = inputs.size()
        x = inputs.view(batch,c,-1)
        x = self.attention_fc(x) + self.bias
        x = x.view(batch,c,1,1)
        x = self.sigmoid(x)
        x = self.dropout(x)
        return inputs * x

class sSEBlock(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.attention_fc = nn.Linear(c,1, bias=False).apply(init_weight)
        self.bias         = nn.Parameter(torch.zeros((1,h,w,1), requires_grad=True))
        self.sigmoid      = nn.Sigmoid()
        
    def forward(self,inputs):
        batch,c,h,w = inputs.size()
        x = torch.transpose(inputs, 1,2) #(*,c,h,w)->(*,h,c,w)
        x = torch.transpose(x, 2,3) #(*,h,c,w)->(*,h,w,c)
        x = self.attention_fc(x) + self.bias
        x = torch.transpose(x, 2,3) #(*,h,w,1)->(*,h,1,w)
        x = torch.transpose(x, 1,2) #(*,h,1,w)->(*,1,h,w)
        x = self.sigmoid(x)
        return inputs * x
    
class scSEBlock(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.cSE = cSEBlock(c,h*w)
        self.sSE = sSEBlock(c,h,w)
    
    def forward(self, inputs):
        x1 = self.cSE(inputs)
        x2 = self.sSE(inputs)
        return x1+x2
    


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta    = nn.utils.spectral_norm(conv1x1(channels, channels//8)).apply(init_weight)
        self.phi      = nn.utils.spectral_norm(conv1x1(channels, channels//8)).apply(init_weight)
        self.g        = nn.utils.spectral_norm(conv1x1(channels, channels//2)).apply(init_weight)
        self.o        = nn.utils.spectral_norm(conv1x1(channels//2, channels)).apply(init_weight)
        self.gamma    = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
    def forward(self, inputs):
        batch,c,h,w = inputs.size()
        theta = self.theta(inputs) #->(*,c/8,h,w)
        phi   = F.max_pool2d(self.phi(inputs), [2,2]) #->(*,c/8,h/2,w/2)
        g     = F.max_pool2d(self.g(inputs), [2,2]) #->(*,c/2,h/2,w/2)
        
        theta = theta.view(batch, self.channels//8, -1) #->(*,c/8,h*w)
        phi   = phi.view(batch, self.channels//8, -1) #->(*,c/8,h*w/4)
        g     = g.view(batch, self.channels//2, -1) #->(*,c/2,h*w/4)
        
        beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1) #->(*,h*w,h*w/4)
        o    = self.o(torch.bmm(g, beta.transpose(1,2)).view(batch,self.channels//2,h,w)) #->(*,c,h,w)
        return self.gamma*o + inputs
    
    

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            conv1x1(in_channel, in_channel//reduction).apply(init_weight),
            nn.ReLU(True),
            conv1x1(in_channel//reduction, in_channel).apply(init_weight)
        )
        
    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x  = torch.sigmoid(x1 + x2)
        return x
    
    
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = conv3x3(2,1).apply(init_weight)
        
    def forward(self, inputs):
        x1,_ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x  = torch.cat([x1,x2], dim=1)
        x  = self.conv3x3(x)
        x  = torch.sigmoid(x)
        return x
    
    
class CBAM(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()
        
    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)
        return x
    
    
class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel).apply(init_weight)
        
    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
        self.bn2 = nn.BatchNorm2d(in_channel).apply(init_weight)
        self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
        self.cbam = CBAM(out_channel, reduction=16)
        self.conv1x1   = conv1x1(in_channel, out_channel).apply(init_weight)
        
    def forward(self, inputs):
        x  = F.relu(self.bn1(inputs))
        x  = self.upsample(x)
        x  = self.conv3x3_1(x)
        x  = self.conv3x3_2(F.relu(self.bn2(x)))
        x  = self.cbam(x)
        x += self.conv1x1(self.upsample(inputs)) #shortcut
        return x
    
    
#U-Net ResNet34 + CBAM + hypercolumns + deepsupervision
class UNET_RESNET34(nn.Module):
    def __init__(self, resolution, deepsupervision, clfhead, load_weights=True):
        super().__init__()
        h,w = resolution
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        
        #encoder
        model_name = 'resnet34' #26M
        resnet34 = pretrainedmodels.__dict__['resnet34'](num_classes=1000,pretrained=None)
        if load_weights:
            resnet34.load_state_dict(torch.load(f'../../../pretrainedmodels_weight/{model_name}.pth'))
        self.conv1   = resnet34.conv1 #(*,3,h,w)->(*,64,h/2,w/2)
        self.bn1     = resnet34.bn1
        self.maxpool = resnet34.maxpool #->(*,64,h/4,w/4)
        self.layer1  = resnet34.layer1 #->(*,64,h/4,w/4) 
        self.layer2  = resnet34.layer2 #->(*,128,h/8,w/8) 
        self.layer3  = resnet34.layer3 #->(*,256,h/16,w/16) 
        self.layer4  = resnet34.layer4 #->(*,512,h/32,w/32) 
        
        #center
        self.center  = CenterBlock(512,512) #->(*,512,h/32,w/32) 
        
        #decoder
        self.decoder4 = DecodeBlock(512+512,64, upsample=True) #->(*,64,h/16,w/16) 
        self.decoder3 = DecodeBlock(64+256,64, upsample=True) #->(*,64,h/8,w/8) 
        self.decoder2 = DecodeBlock(64+128,64,  upsample=True) #->(*,64,h/4,w/4) 
        self.decoder1 = DecodeBlock(64+64,64,   upsample=True) #->(*,64,h/2,w/2)
        self.decoder0 = DecodeBlock(64,64, upsample=True) #->(*,64,h,w) 
        
        #upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #deep supervision
        self.deep4 = conv1x1(64,1).apply(init_weight)
        self.deep3 = conv1x1(64,1).apply(init_weight)
        self.deep2 = conv1x1(64,1).apply(init_weight)
        self.deep1 = conv1x1(64,1).apply(init_weight)
        
        #final conv
        self.final_conv = nn.Sequential(
            conv3x3(320,64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64,1).apply(init_weight)
        )
        
        #clf head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(
            nn.BatchNorm1d(512).apply(init_weight),
            nn.Linear(512,512).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(512).apply(init_weight),
            nn.Linear(512,1).apply(init_weight)
        )
        
    def forward(self, inputs):
        #encoder
        x0 = F.relu(self.bn1(self.conv1(inputs))) #->(*,64,h/2,w/2) 
        x0 = self.maxpool(x0) #->(*,64,h/4,w/4)
        x1 = self.layer1(x0) #->(*,64,h/4,w/4)
        x2 = self.layer2(x1) #->(*,128,h/8,w/8)
        x3 = self.layer3(x2) #->(*,256,h/16,w/16)
        x4 = self.layer4(x3) #->(*,512,h/32,w/32)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        if config['clf_threshold'] is not None:
            if (torch.sigmoid(logits_clf)>config['clf_threshold']).sum().item()==0:
                bs,_,h,w = inputs.shape
                logits = torch.zeros((bs,1,h,w))
                if self.clfhead:
                    if self.deepsupervision:
                        return logits,_,_
                    else:
                        return logits,_
                else:
                    if self.deepsupervision:
                        return logits,_
                    else:
                        return logits
        
        #center
        y5 = self.center(x4) #->(*,512,h/32,w/32)
        
        #decoder
        y4 = self.decoder4(torch.cat([x4,y5], dim=1)) #->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3,y4], dim=1)) #->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2,y3], dim=1)) #->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1,y2], dim=1)) #->(*,64,h/2,w/2)
        y0 = self.decoder0(y1) #->(*,64,h,w)
        
        #hypercolumns
        y4 = self.upsample4(y4) #->(*,64,h,w)
        y3 = self.upsample3(y3) #->(*,64,h,w)
        y2 = self.upsample2(y2) #->(*,64,h,w)
        y1 = self.upsample1(y1) #->(*,64,h,w)
        hypercol = torch.cat([y0,y1,y2,y3,y4], dim=1)
        
        #final conv
        logits = self.final_conv(hypercol) #->(*,1,h,w)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        
        if self.clfhead:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps, logits_clf
            else:
                return logits, logits_clf
        else:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps
            else:
                return logits

        
#U-Net SeResNext50 + CBAM + hypercolumns + deepsupervision
class UNET_SERESNEXT50(nn.Module):
    def __init__(self, resolution, deepsupervision, clfhead, load_weights=True):
        super().__init__()
        h,w = resolution
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        
        #encoder
        model_name = 'se_resnext50_32x4d' #26M
        seresnext50 = pretrainedmodels.__dict__[model_name](pretrained=None)
        if load_weights:
            seresnext50.load_state_dict(torch.load(f'../../../pretrainedmodels_weight/{model_name}.pth'))
        
        self.encoder0 = nn.Sequential(
            seresnext50.layer0.conv1, #(*,3,h,w)->(*,64,h/2,w/2)
            seresnext50.layer0.bn1,
            seresnext50.layer0.relu1,
        )
        self.encoder1 = nn.Sequential(
            seresnext50.layer0.pool, #->(*,64,h/4,w/4)
            seresnext50.layer1 #->(*,256,h/4,w/4)
        )
        self.encoder2 = seresnext50.layer2 #->(*,512,h/8,w/8)
        self.encoder3 = seresnext50.layer3 #->(*,1024,h/16,w/16)
        self.encoder4 = seresnext50.layer4 #->(*,2048,h/32,w/32)
        
        #center
        self.center  = CenterBlock(2048,512) #->(*,512,h/32,w/32) 10,16
        
        #decoder
        self.decoder4 = DecodeBlock(512+2048,64, upsample=True) #->(*,64,h/16,w/16) 20,32
        self.decoder3 = DecodeBlock(64+1024,64, upsample=True) #->(*,64,h/8,w/8) 40,64
        self.decoder2 = DecodeBlock(64+512,64,  upsample=True) #->(*,64,h/4,w/4) 80,128
        self.decoder1 = DecodeBlock(64+256,64,   upsample=True) #->(*,64,h/2,w/2) 160,256
        self.decoder0 = DecodeBlock(64,64, upsample=True) #->(*,64,h,w) 320,512
        
        #upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #deep supervision
        self.deep4 = conv1x1(64,1).apply(init_weight)
        self.deep3 = conv1x1(64,1).apply(init_weight)
        self.deep2 = conv1x1(64,1).apply(init_weight)
        self.deep1 = conv1x1(64,1).apply(init_weight)
        
        #final conv
        self.final_conv = nn.Sequential(
            conv3x3(320,64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64,1).apply(init_weight)
        )
        
        #clf head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(
            nn.BatchNorm1d(2048).apply(init_weight),
            nn.Linear(2048,512).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(512).apply(init_weight),
            nn.Linear(512,1).apply(init_weight)
        )
        
    def forward(self, inputs):
        #encoder
        x0 = self.encoder0(inputs) #->(*,64,h/2,w/2) 160,256
        x1 = self.encoder1(x0) #->(*,256,h/4,w/4)
        x2 = self.encoder2(x1) #->(*,512,h/8,w/8)
        x3 = self.encoder3(x2) #->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3) #->(*,2048,h/32,w/32)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        if config['clf_threshold'] is not None:
            if (torch.sigmoid(logits_clf)>config['clf_threshold']).sum().item()==0:
                bs,_,h,w = inputs.shape
                logits = torch.zeros((bs,1,h,w))
                if self.clfhead:
                    if self.deepsupervision:
                        return logits,_,_
                    else:
                        return logits,_
                else:
                    if self.deepsupervision:
                        return logits,_
                    else:
                        return logits
        
        #center
        y5 = self.center(x4) #->(*,320,h/32,w/32)
        
        #decoder
        y4 = self.decoder4(torch.cat([x4,y5], dim=1)) #->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3,y4], dim=1)) #->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2,y3], dim=1)) #->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1,y2], dim=1)) #->(*,64,h/2,w/2) 160,256
        y0 = self.decoder0(y1) #->(*,64,h,w) 320,512
        
        #hypercolumns
        y4 = self.upsample4(y4) #->(*,64,h,w)
        y3 = self.upsample3(y3) #->(*,64,h,w)
        y2 = self.upsample2(y2) #->(*,64,h,w)
        y1 = self.upsample1(y1) #->(*,64,h,w)
        hypercol = torch.cat([y0,y1,y2,y3,y4], dim=1)
        
        #final conv
        logits = self.final_conv(hypercol) #->(*,4,h,w)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        
        if self.clfhead:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps, logits_clf
            else:
                return logits, logits_clf
        else:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps
            else:
                return logits
    

#U-Net SeResNext101 + CBAM + hypercolumns + deepsupervision
class UNET_SERESNEXT101(nn.Module):
    def __init__(self, resolution, deepsupervision, clfhead, load_weights=True):
        super().__init__()
        h,w = resolution
        self.deepsupervision = deepsupervision
        self.clfhead = clfhead
        
        #encoder
        model_name = 'se_resnext101_32x4d'
        seresnext101 = pretrainedmodels.__dict__[model_name](pretrained=None)
        if load_weights:
            seresnext101.load_state_dict(torch.load(f'../../../pretrainedmodels_weight/{model_name}.pth'))
        
        self.encoder0 = nn.Sequential(
            seresnext101.layer0.conv1, #(*,3,h,w)->(*,64,h/2,w/2)
            seresnext101.layer0.bn1,
            seresnext101.layer0.relu1,
        )
        self.encoder1 = nn.Sequential(
            seresnext101.layer0.pool, #->(*,64,h/4,w/4)
            seresnext101.layer1 #->(*,256,h/4,w/4)
        )
        self.encoder2 = seresnext101.layer2 #->(*,512,h/8,w/8)
        self.encoder3 = seresnext101.layer3 #->(*,1024,h/16,w/16)
        self.encoder4 = seresnext101.layer4 #->(*,2048,h/32,w/32)
        
        #center
        self.center  = CenterBlock(2048,512) #->(*,512,h/32,w/32)
        
        #decoder
        self.decoder4 = DecodeBlock(512+2048,64, upsample=True) #->(*,64,h/16,w/16)
        self.decoder3 = DecodeBlock(64+1024,64, upsample=True) #->(*,64,h/8,w/8)
        self.decoder2 = DecodeBlock(64+512,64,  upsample=True) #->(*,64,h/4,w/4) 
        self.decoder1 = DecodeBlock(64+256,64,   upsample=True) #->(*,64,h/2,w/2) 
        self.decoder0 = DecodeBlock(64,64, upsample=True) #->(*,64,h,w) 
        
        #upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #deep supervision
        self.deep4 = conv1x1(64,1).apply(init_weight)
        self.deep3 = conv1x1(64,1).apply(init_weight)
        self.deep2 = conv1x1(64,1).apply(init_weight)
        self.deep1 = conv1x1(64,1).apply(init_weight)
        
        #final conv
        self.final_conv = nn.Sequential(
            conv3x3(320,64).apply(init_weight),
            nn.ELU(True),
            conv1x1(64,1).apply(init_weight)
        )
        
        #clf head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(
            nn.BatchNorm1d(2048).apply(init_weight),
            nn.Linear(2048,512).apply(init_weight),
            nn.ELU(True),
            nn.BatchNorm1d(512).apply(init_weight),
            nn.Linear(512,1).apply(init_weight)
        )
        
    def forward(self, inputs):
        #encoder
        x0 = self.encoder0(inputs) #->(*,64,h/2,w/2)
        x1 = self.encoder1(x0) #->(*,256,h/4,w/4)
        x2 = self.encoder2(x1) #->(*,512,h/8,w/8)
        x3 = self.encoder3(x2) #->(*,1024,h/16,w/16)
        x4 = self.encoder4(x3) #->(*,2048,h/32,w/32)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        if config['clf_threshold'] is not None:
            if (torch.sigmoid(logits_clf)>config['clf_threshold']).sum().item()==0:
                bs,_,h,w = inputs.shape
                logits = torch.zeros((bs,1,h,w))
                if self.clfhead:
                    if self.deepsupervision:
                        return logits,_,_
                    else:
                        return logits,_
                else:
                    if self.deepsupervision:
                        return logits,_
                    else:
                        return logits
        
        #center
        y5 = self.center(x4) #->(*,320,h/32,w/32)
        
        #decoder
        y4 = self.decoder4(torch.cat([x4,y5], dim=1)) #->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3,y4], dim=1)) #->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2,y3], dim=1)) #->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1,y2], dim=1)) #->(*,64,h/2,w/2) 
        y0 = self.decoder0(y1) #->(*,64,h,w)
        
        #hypercolumns
        y4 = self.upsample4(y4) #->(*,64,h,w)
        y3 = self.upsample3(y3) #->(*,64,h,w)
        y2 = self.upsample2(y2) #->(*,64,h,w)
        y1 = self.upsample1(y1) #->(*,64,h,w)
        hypercol = torch.cat([y0,y1,y2,y3,y4], dim=1)
        
        #final conv
        logits = self.final_conv(hypercol) #->(*,1,h,w)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)
        
        if self.clfhead:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps, logits_clf
            else:
                return logits, logits_clf
        else:
            if self.deepsupervision:
                s4 = self.deep4(y4)
                s3 = self.deep3(y3)
                s2 = self.deep2(y2)
                s1 = self.deep1(y1)
                logits_deeps = [s4,s3,s2,s1]
                return logits, logits_deeps
            else:
                return logits    

    
def build_model(resolution, deepsupervision, clfhead, load_weights):
    model_name = config['model_name']
    if model_name=='resnet34':
        model = UNET_RESNET34(resolution, deepsupervision, clfhead, load_weights)
    elif model_name=='seresnext50':
        model = UNET_SERESNEXT50(resolution, deepsupervision, clfhead, load_weights)
    elif model_name=='seresnext101':
        model = UNET_SERESNEXT101(resolution, deepsupervision, clfhead, load_weights)
    return model

# %% [markdown] {"papermill":{"duration":0.016345,"end_time":"2021-03-31T10:37:33.38936","exception":false,"start_time":"2021-03-31T10:37:33.373015","status":"completed"},"tags":[]}
# # Inference

# %% [code] {"papermill":{"duration":34.271842,"end_time":"2021-03-31T10:38:07.677397","exception":false,"start_time":"2021-03-31T10:37:33.405555","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:45:37.724632Z","iopub.execute_input":"2022-06-28T19:45:37.724969Z","iopub.status.idle":"2022-06-28T19:46:11.349732Z","shell.execute_reply.started":"2022-06-28T19:45:37.724935Z","shell.execute_reply":"2022-06-28T19:46:11.348936Z"}}
#from models import build_model

LOAD_LOCAL_WEIGHT_PATH_LIST = {}
for seed in config['split_seed_list']:
    LOAD_LOCAL_WEIGHT_PATH_LIST[seed] = []
    for fold in config['FOLD_LIST']:
        LOAD_LOCAL_WEIGHT_PATH_LIST[seed].append(opj(config['model_path'],f'model_seed{seed}_fold{fold}_bestscore.pth'))
        #LOAD_LOCAL_WEIGHT_PATH_LIST[seed].append(opj(config['model_path'],f'model_seed{seed}_fold{fold}_swa.pth'))

model_list = {}
for seed in config['split_seed_list']:
    model_list[seed] = []
    for path in LOAD_LOCAL_WEIGHT_PATH_LIST[seed]:
        print("Loading weights from %s" % path)
        
        model = build_model(resolution=(None,None), #config['resolution'], 
                            deepsupervision=config['deepsupervision'], 
                            clfhead=config['clfhead'],
                            load_weights=False).to(device)
        
        model.load_state_dict(torch.load(path))
        model.eval()
        model_list[seed].append(model) 

# %% [code] {"papermill":{"duration":1.342372,"end_time":"2021-03-31T10:38:09.046504","exception":false,"start_time":"2021-03-31T10:38:07.704132","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:46:18.124774Z","iopub.execute_input":"2022-06-28T19:46:18.125305Z","iopub.status.idle":"2022-06-28T19:46:19.394826Z","shell.execute_reply.started":"2022-06-28T19:46:18.125243Z","shell.execute_reply":"2022-06-28T19:46:19.393967Z"}}
import numpy as np
from albumentations import (Compose, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, ElasticTransform,
                            GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
                            RandomBrightnessContrast, HueSaturationValue, IAASharpen,
                            RandomGamma, RandomBrightness, RandomBrightnessContrast,
                            GaussianBlur,CLAHE,
                            Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
                            Normalize, OneOf, NoOp)
from albumentations.pytorch import ToTensor, ToTensorV2
#from get_config import *
#config = get_config()

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def get_transforms_test():
    transforms = Compose([
        Normalize(mean=(MEAN[0], MEAN[1], MEAN[2]), 
                  std=(STD[0], STD[1], STD[2])),
        ToTensorV2(),
    ] )
    return transforms

def denormalize(z, mean=MEAN.reshape(-1,1,1), std=STD.reshape(-1,1,1)):
    return std*z + mean

# %% [code] {"papermill":{"duration":0.042755,"end_time":"2021-03-31T10:38:09.106626","exception":false,"start_time":"2021-03-31T10:38:09.063871","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:46:22.003064Z","iopub.execute_input":"2022-06-28T19:46:22.003417Z","iopub.status.idle":"2022-06-28T19:46:22.026564Z","shell.execute_reply.started":"2022-06-28T19:46:22.003383Z","shell.execute_reply":"2022-06-28T19:46:22.025699Z"}}
from torch.utils.data import Dataset

class HuBMAPDataset(Dataset):
    def __init__(self, idx, df):
        super().__init__()
        filename = df.loc[idx, 'id']+'.tiff'
        path = opj(INPUT_PATH,'test',filename)
        self.data = rasterio.open(path)
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i,subdataset in enumerate(subdatasets,0):
                    self.layers.append(rasterio.open(subdataset))
        self.h, self.w = self.data.height, self.data.width
        self.input_sz = config['input_resolution']
        self.sz = config['resolution']
        self.pad_sz = config['pad_size'] # add to each input tile
        self.pred_sz = self.sz - 2*self.pad_sz
        self.pad_h = self.pred_sz - self.h % self.pred_sz # add to whole slide
        self.pad_w = self.pred_sz - self.w % self.pred_sz # add to whole slide
        self.num_h = (self.h + self.pad_h) // self.pred_sz
        self.num_w = (self.w + self.pad_w) // self.pred_sz
        self.transforms = get_transforms_test()
        
    def __len__(self):
        return self.num_h * self.num_w
    
    def __getitem__(self, idx): # idx = i_h * self.num_w + i_w
        # prepare coordinates for rasterio
        i_h = idx // self.num_w
        i_w = idx % self.num_w
        y = i_h*self.pred_sz 
        x = i_w*self.pred_sz
        py0,py1 = max(0,y), min(y+self.pred_sz, self.h)
        px0,px1 = max(0,x), min(x+self.pred_sz, self.w)
        
        # padding coordinate for rasterio
        qy0,qy1 = max(0,y-self.pad_sz), min(y+self.pred_sz+self.pad_sz, self.h)
        qx0,qx1 = max(0,x-self.pad_sz), min(x+self.pred_sz+self.pad_sz, self.w)
        
        # placeholder for input tile (before resize)
        img = np.zeros((self.sz,self.sz,3), np.uint8)
        
        # replace the value
        if self.data.count == 3:
            img[0:qy1-qy0, 0:qx1-qx0] =\
                np.moveaxis(self.data.read([1,2,3], window=Window.from_slices((qy0,qy1),(qx0,qx1))), 0,-1)
        else:
            for i,layer in enumerate(self.layers):
                img[0:qy1-qy0, 0:qx1-qx0, i] =\
                    layer.read(1,window=Window.from_slices((qy0,qy1),(qx0,qx1)))
        if self.sz != self.input_sz:
            img = cv2.resize(img, (self.input_sz, self.input_sz), interpolation=cv2.INTER_AREA)
        img = self.transforms(image=img)['image'] # to normalized tensor
        return {'img':img, 'p':[py0,py1,px0,px1], 'q':[qy0,qy1,qx0,qx1]}

# %% [code] {"papermill":{"duration":0.048072,"end_time":"2021-03-31T10:38:09.171422","exception":false,"start_time":"2021-03-31T10:38:09.12335","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:46:25.626997Z","iopub.execute_input":"2022-06-28T19:46:25.627349Z","iopub.status.idle":"2022-06-28T19:46:25.658920Z","shell.execute_reply.started":"2022-06-28T19:46:25.627316Z","shell.execute_reply":"2022-06-28T19:46:25.657947Z"}}
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import gc
import math


def my_collate_fn(batch):
    img = []
    p = []
    q = []
    for sample in batch:
        img.append(sample['img'])
        p.append(sample['p'])
        q.append(sample['q'])
    img = torch.stack(img)
    return {'img':img, 'p':p, 'q':q}


seed = 0

def get_pred_mask(idx, df, model_list):
    ds = HuBMAPDataset(idx, df)
    print("h and w ", ds.h,"   ",ds.w)
    #rasterio cannot be used with multiple workers
    dl = DataLoader(ds,batch_size=config['test_batch_size'],
                    num_workers=0,shuffle=False,pin_memory=True,
                    collate_fn=my_collate_fn) 
    
    pred_mask = np.zeros((len(ds),ds.pred_sz,ds.pred_sz), dtype=np.uint8)
    
    i_data = 0
    for data in tqdm(dl):
        bs = data['img'].shape[0]
        img_patch = data['img'] # (bs,3,input_res,input_res)
        pred_mask_float = 0
        for model in model_list[seed]:
            with torch.no_grad():
                if config['tta']>0:
                    pred_mask_float += torch.sigmoid(model(img_patch.to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                if config['tta']>1:
                    # h-flip
                    _pred_mask_float = torch.sigmoid(model(img_patch.flip([-1]).to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                    pred_mask_float += _pred_mask_float[:,:,::-1]
                if config['tta']>2:
                    # v-flip
                    _pred_mask_float = torch.sigmoid(model(img_patch.flip([-2]).to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                    pred_mask_float += _pred_mask_float[:,::-1,:]
                if config['tta']>3:
                    # h-v-flip
                    _pred_mask_float = torch.sigmoid(model(img_patch.flip([-1,-2]).to(device, torch.float32, non_blocking=True))).detach().cpu().numpy()[:,0,:,:] #.squeeze()
                    pred_mask_float += _pred_mask_float[:,::-1,::-1]
        pred_mask_float = pred_mask_float / min(config['tta'],4) / len(model_list[seed]) # (bs,input_res,input_res)
        
        # resize
        pred_mask_float = np.vstack([cv2.resize(_mask.astype(np.float32), (ds.sz,ds.sz))[None] for _mask in pred_mask_float])
        
        # float to uint8
        pred_mask_int = (pred_mask_float>config['mask_threshold']).astype(np.uint8)
        
        # replace the values
        for j in range(bs):
            py0,py1,px0,px1 = data['p'][j]
            qy0,qy1,qx0,qx1 = data['q'][j]
            pred_mask[i_data+j,0:py1-py0, 0:px1-px0] = pred_mask_int[j, py0-qy0:py1-qy0, px0-qx0:px1-qx0] # (pred_sz,pred_sz)
        i_data += bs
    
    pred_mask = pred_mask.reshape(ds.num_h*ds.num_w, ds.pred_sz, ds.pred_sz).reshape(ds.num_h, ds.num_w, ds.pred_sz, ds.pred_sz)
    pred_mask = pred_mask.transpose(0,2,1,3).reshape(ds.num_h*ds.pred_sz, ds.num_w*ds.pred_sz)
    pred_mask = pred_mask[:ds.h,:ds.w] # back to the original slide size
    non_zero_ratio = (pred_mask).sum() / (ds.h*ds.w)
    print('non_zero_ratio = {:.4f}'.format(non_zero_ratio))
    return pred_mask,ds.h,ds.w

def get_rle(y_preds, h,w):
    rle = mask2rle(y_preds, shape=(h,w), small_mask_threshold=config['small_mask_threshold'])
    return rle

# %% [code] {"papermill":{"duration":1850.195536,"end_time":"2021-03-31T11:08:59.383643","exception":false,"start_time":"2021-03-31T10:38:09.188107","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-06-28T19:46:29.541363Z","iopub.execute_input":"2022-06-28T19:46:29.541698Z","iopub.status.idle":"2022-06-28T19:47:16.341862Z","shell.execute_reply.started":"2022-06-28T19:46:29.541666Z","shell.execute_reply":"2022-06-28T19:47:16.341013Z"}}
#%%time
print("len df: ", len(sub_df))
t_start = time.time()
for idx in range(len(sub_df)): 
    print('idx = ', idx)
    pred_mask,h,w = get_pred_mask(idx, sub_df, model_list)
    print("h and w ", h,"   ",w)
    rle = get_rle(pred_mask,h,w)
    sub_df.loc[idx,'predicted'] = rle

print("time: ", elapsed_time(t_start))
# %% [markdown] {"papermill":{"duration":0.01885,"end_time":"2021-03-31T11:08:59.422121","exception":false,"start_time":"2021-03-31T11:08:59.403271","status":"completed"},"tags":[]}
# # Submission

# %% [code] {"execution":{"iopub.execute_input":"2021-03-31T11:08:59.466298Z","iopub.status.busy":"2021-03-31T11:08:59.465444Z","iopub.status.idle":"2021-03-31T11:08:59.729766Z","shell.execute_reply":"2021-03-31T11:08:59.728687Z"},"papermill":{"duration":0.288472,"end_time":"2021-03-31T11:08:59.729891","exception":false,"start_time":"2021-03-31T11:08:59.441419","status":"completed"},"tags":[]}
sub_df.to_csv('submission.csv', index=False)

# %% [code] {"execution":{"iopub.execute_input":"2021-03-31T11:08:59.781668Z","iopub.status.busy":"2021-03-31T11:08:59.780794Z","iopub.status.idle":"2021-03-31T11:08:59.792171Z","shell.execute_reply":"2021-03-31T11:08:59.791751Z"},"papermill":{"duration":0.043112,"end_time":"2021-03-31T11:08:59.792271","exception":false,"start_time":"2021-03-31T11:08:59.749159","status":"completed"},"tags":[]}
sub_df

# %% [code] {"papermill":{"duration":0.019351,"end_time":"2021-03-31T11:08:59.831375","exception":false,"start_time":"2021-03-31T11:08:59.812024","status":"completed"},"tags":[]}
def dice_score(A,B, smooth = 0):

    if (A==0).all() and (B==0).all(): return 1.0

    return 2*((A*B).sum())/(A.sum()+B.sum()+smooth)