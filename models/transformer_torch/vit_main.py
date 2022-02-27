# main file for vision transformer
# Created: 6/16/2021
# Status: in progress
# CUBLAS_WORKSPACE_CONFIG=:4096:8 python vit_main.py

import sys
import torch

import numpy as np

from networks import ViT_Wrapper
from utils import read_json


def ViT(num_exps, model_name, config, Wrapper):
    print('Evaluation metric: {}'.format(config['metric']))
    c_te, c_tr = [], []
    for exp_idx in range(num_exps):
        print('*'*50)
        vit = Wrapper(config          = config,
                      exp_idx         = exp_idx,
                      num_fold        = num_exps,
                      seed            = 1000*exp_idx,
                      model_name      = model_name)
        # cnn.load('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0), fixed=False)
        vit.train(epochs = config['train_epochs'])
        vit.result()
        sys.exit()
        # cnn.check('./checkpoint_dir/{}_exp{}/'.format('cnn_mri_pre', 0))
        # cnn.shap()
        # print('exit')
    # print('CI train: %.3f+-%.3f' % (np.mean(c_tr), np.std(c_tr)))


def main():
    torch.use_deterministic_algorithms(True)
    print('-'*100)
    print('Running vision transformer (ViT)')
    config = read_json('./config.json')['vit']
    num_exps = 1
    model_name = 'vit_{}'.format(config['metric'])
    ViT(num_exps, model_name, config, ViT_Wrapper)
    print('-'*100)
    print('OK')


if __name__ == "__main__":
    main()
