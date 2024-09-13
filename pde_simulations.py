import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import gstools as gs
from pde import DiffusionPDE, ScalarField, UnitGrid

from neuralop.models import FNO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import scipy.linalg as sla
import scipy.optimize as sopt
from pyswarms.single import GlobalBestPSO

from sklearn.metrics import r2_score, mean_absolute_percentage_error
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

NSAMPLES  = 500
NDIM      = 128

def check_torch(verbose:bool=True):
    if torch.cuda.is_available():
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('# Device(s) available: {}, Name(s): {}'.format(count, name))
            print('-'*60)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
    else:
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('-'*60)
        device = torch.device('cpu')
        return device

DEVICE = check_torch()

#############################################################################
################################ MODEL SPACE ################################
#############################################################################
def make_model_space(variance=1, len_scale=40, showfig:bool=True, figsize=(15,6), cmap='jet'):
    x = y = range(NDIM)
    model = gs.Gaussian(dim=2, var=variance, len_scale=len_scale)

    mm = np.zeros((NSAMPLES, NDIM, NDIM))
    for i in tqdm(range(NSAMPLES), desc='Generating model space'):
        srf = gs.SRF(model)
        mm[i] = srf((x,y), mesh_type='structured')
    np.save('features.npy', mm)

    if showfig:
        plt.figure(figsize=figsize)
        for i in range(36):
            plt.subplot(3, 12, i+1)
            plt.imshow(mm[i], cmap=cmap)
            plt.title('R{}'.format(i))
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('figures/features_realizations.png', dpi=600)
        plt.close()

    return mm

def load_model_space(fname:str='features.npy'):
    return np.load(fname)

#############################################################################
################################ DATA SPACE #################################
#############################################################################
def make_data_space(mm, monitor=50, showfig:bool=True, figsize=(15,5), cmap='jet',
                    diffusivity=(2/3), bcs=[{'value':0},{'value':0}], noise=0.0, t_range=200, dt=0.5,
                    backend='numpy', scheme='rk', ret_info=False, trackers=[None]):
    dd = np.zeros_like(mm)
    grid = UnitGrid([mm.shape[1],mm.shape[2]])
    state = ScalarField.random_uniform(grid, 0.2, 0.3)

    start = time()
    for i in range(mm.shape[0]):
        state.data = mm[i]
        eq = DiffusionPDE(diffusivity=diffusivity, bc=bcs, noise=noise)
        dd[i] = eq.solve(state, t_range=t_range, dt=dt, backend=backend, scheme=scheme, ret_info=ret_info, tracker=trackers).data
        if (i+1) % monitor == 0:
            print('Simulation [{}/{}] done ..'.format(i+1, mm.shape[0]))
    print('Total Simulation Time: {:.2f}'.format((time()-start)/60))
    np.save('targets.npy', dd)

    if showfig:
        labs = ['LogPerm','Diffusion']
        hues = ['black','blue']
        mult = 5
        fig, axs = plt.subplots(2, 12, figsize=figsize, sharex=True, sharey=True)
        for j in range(12):
            k = j*mult
            ax1, ax2 = axs[0,j], axs[1,j]
            im1 = ax1.imshow(mm[k], cmap=cmap)
            im2 = ax2.imshow(dd[k], cmap=cmap)
            ax1.set(title='R{}'.format(k))
            [a.set_ylabel(labs[i], color=hues[i]) for i,a in enumerate([ax1,ax2])] if j==0 else None
        plt.tight_layout()
        plt.savefig('figures/features_and_targets.png', dpi=600)
        plt.close()

    return dd

def load_data_space(fname:str='targets.npy'):
    return np.load(fname)

if __name__ == '__main__':
    mm = load_model_space()
    dd = make_data_space(mm)
    print('Model Space Shape: {}'.format(mm.shape))
    print('Data Space Shape: {}'.format(dd.shape))