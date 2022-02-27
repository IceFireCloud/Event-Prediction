# network wrappers for vision transformer
# Created: 6/16/2021
# Status: in progress

import torch
import os, sys
import glob
import math
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored, integrated_brier_score

from scipy import interpolate
from dataloader import ViT_Data
from models import _ViT_Model

def cus_loss(preds, obss, hits, bins=torch.Tensor([[0, 24, 48, 108]])):
    if torch.cuda.is_available():
        bins = bins.cuda()
    bin_centers = (bins[0, 1:] + bins[0, :-1])/2
    survived_bins_censored = torch.ge(torch.mul(obss.view(-1, 1),1-hits.view(-1,1)), bin_centers)
    survived_bins_hits = torch.ge(torch.mul(obss.view(-1,1), hits.view(-1,1)), bins[0,1:])
    survived_bins = torch.logical_or(survived_bins_censored, survived_bins_hits)
    survived_bins = torch.where(survived_bins, 1, 0)
    event_bins = torch.logical_and(torch.ge(obss.view(-1, 1), bins[0, :-1]), torch.lt(obss.view(-1, 1), bins[0, 1:]))
    event_bins = torch.where(event_bins, 1, 0)
    hit_bins = torch.mul(event_bins, hits.view(-1, 1))
    l_h_x = 1+survived_bins*(preds-1)
    n_l_h_x = 1-hit_bins*preds
    cat_tensor = torch.cat((l_h_x, n_l_h_x), axis=0)
    total = -torch.log(torch.clamp(cat_tensor, min=1e-12))
    pos_sum = torch.sum(total)
    neg_sum = torch.sum(pos_sum)
    return neg_sum

class ViT_Wrapper:
    def __init__(self, config, exp_idx, num_fold, seed, model_name):
        self.gpu = 1

        self.config = config
        self.lr = config['lr']
        self.exp_idx = exp_idx
        self.num_fold = num_fold
        self.seed = seed
        self.model_name = model_name
        self.metric = config['metric']
        self.checkpoint_dir = './checkpoint_dir/{}_exp{}/'.format(self.model_name, exp_idx)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        torch.manual_seed(seed)
        self.prepare_dataloader(config['batch_size'])

        # in_size = 16*16*168
        vector_len = config['out_dim']
        self.targets = list(range(vector_len))
        self.model = _ViT_Model(config).cuda()
        if self.gpu != 1:
            self.model = self.model.cpu()

        if self.metric == 'Standard':
            # self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.imbalanced_ratio)).cuda()
            # self.criterion = nn.CrossEntropyLoss().cuda()
            self.criterion = nn.MSELoss().cuda()
        else:
            self.criterion = sur_loss
        self.optimizer = optim.SGD(self.model.parameters(), lr=config['lr'], weight_decay=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)

    def prepare_dataloader(self, batch_size):
        train_data = ViT_Data(self.exp_idx, stage='train', seed=self.seed, name=self.model_name, fold=self.num_fold)
        # sample_weight, self.imbalanced_ratio = train_data.get_sample_weights()
        self.train_data = train_data
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
        if self.gpu != 1:
            self.train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)

        valid_data = ViT_Data(self.exp_idx, stage='valid', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)

        test_data  = ViT_Data(self.exp_idx, stage='test', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.test_data = test_data
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

        all_data = ViT_Data(self.exp_idx, stage='all', seed=self.seed, name=self.model_name, fold=self.num_fold)
        self.all_data = all_data
        self.all_dataloader = DataLoader(all_data, batch_size=len(all_data))

    def load(self, dir, fixed=False):
        # need to update
        print('not implemented')
        # print('loading pre-trained model...')
        # dir = glob.glob(dir + '*.pth')
        # st = torch.load(dir[0])
        # del st['l2.weight']
        # del st['l2.bias']
        # self.model.load_state_dict(st, strict=False)
        # if fixed:
        #     ps = []
        #     for n, p in self.model.named_parameters():
        #         if n == 'l2.weight' or n == 'l2.bias' or n == 'l1.weight' or n == 'l1.bias':
        #             ps += [p]
        #             # continue
        #         else:
        #             pass
        #             p.requires_grad = False
        #     self.optimizer = optim.SGD(ps, lr=self.lr, weight_decay=0.01)
        # # for n, p in self.model.named_parameters():
        #     # print(n, p.requires_grad)
        # print('loaded.')

    def train(self, epochs):
        print('training...')
        self.optimal_valid_metric = np.inf
        self.optimal_epoch        = -1

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for self.epoch in range(epochs):
            train_loss = self.train_model_epoch()
            if self.epoch % 10 == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)
            if self.epoch % (epochs//10) == 0:
                val_loss = self.valid_model_epoch()
                self.save_checkpoint(val_loss)

                end.record()
                torch.cuda.synchronize()

                print('{}th epoch validation loss [{}] ='.format(self.epoch, self.config['metric']), '%.3f' % (math.sqrt(val_loss)), '|| train_loss = %.3f' % (math.sqrt(train_loss)), '|| time(s) =', start.elapsed_time(end)//1000)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric.item())
        print('Location: {}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
        return self.optimal_valid_metric

    def train_model_epoch(self):
        self.model.train(True)
        total_loss = []
        for inputs, targets in self.train_dataloader:
            inputs, targets = inputs.float(), targets.float()
            inputs, targets = inputs.cuda(), targets.cuda()
            if self.gpu != 1:
                inputs, targets = inputs.cpu(), targets.cpu()
            self.model.zero_grad()
            preds = self.model(inputs)
            # print(targets.shape)
            # sys.exit()

            if self.metric == 'Standard':
                loss = self.criterion(preds, targets)
            else:
                loss = self.criterion(preds, obss, hits)
            total_loss += [loss.item()]

            # torch.use_deterministic_algorithms(False)
            loss.backward()
            # torch.use_deterministic_algorithms(True)
            # clip = 1
            # nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            # print('hits', hits[:10])
        return np.mean(total_loss)
        
    def result(self):
        # provides eval results
        print('generating results...')
        dir = glob.glob(self.checkpoint_dir + '*.pth')
        self.model.load_state_dict(torch.load(dir[0]))
        self.model.train(False)
        with torch.no_grad():
            preds_all = []
            targets_all = []
            for inputs, targets in self.valid_dataloader:
                # here only use 1 patch
                inputs, targets = inputs.float(), targets.float()
                if self.gpu == 1:
                    inputs, targets = inputs.cuda(), targets.cuda()
                preds_all += [self.model(inputs).cpu().numpy().squeeze()]
                targets_all += [targets.squeeze().tolist()]
            if self.gpu == 1:
                preds_all, targets_all = torch.tensor(preds_all).cuda(), torch.tensor(targets_all).cuda()
            else:
                preds_all, targets_all = torch.tensor(preds_all), torch.tensor(targets_all)

            if self.metric == 'Standard':
                loss = self.criterion(preds_all, targets_all)
            else:
                loss = self.criterion(preds_all, obss_all, hits_all)
        print('RMSE', math.sqrt(loss))
        for i in range(targets_all.shape[0]):
            if sum(targets_all[i]) != 0:
                break
                print(torch.round(preds_all[i]))
                print(targets_all[i])
                print(torch.round(preds_all[i]) == targets_all[i])
        # def draw(filename_s, avg, single_file=False, draw_trend=False):
        filename_s = './figure/'
        for loc in range(256):
            pred = torch.round(preds_all)[:,loc].cpu().numpy()
            truth = targets_all[:,loc].cpu().numpy()
            # truth = df['truth'].values
            fig, ax = plt.subplots()
            plt.plot(pred,  'r+-', label='prediction', markersize=5, linestyle='None')
            plt.plot(truth, 'ko-', label='truth', markersize=0, linewidth=1)
            ax.set(xlabel='time (hour)', ylabel='# of crimes', title='result')
            ax.grid()
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

            plt.savefig(filename_s+str(loc)+'.png')
            plt.close()

            if 1:
                pred = np.cumsum(pred)
                truth = np.cumsum(truth)
                avg = 0.09558517284464807
                avg = np.cumsum(np.full(truth.shape, avg))
                fig, ax = plt.subplots()
                plt.plot(pred,  'r+-', label='prediction', markersize=5, linestyle='None')
                plt.plot(avg,  'b+-', label='average', markersize=5, linestyle='None')
                plt.plot(truth, 'ko-', label='truth', markersize=0, linewidth=1)
                ax.set(xlabel='time (hour)', ylabel='total # of crimes', title='result')
                ax.grid()
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

                plt.savefig(filename_s+str(loc)+'_cumsum.png')
                plt.close()

        print('figure drawn')

    def valid_model_epoch(self):
        with torch.no_grad():
            self.model.train(False)
            preds_all = []
            targets_all = []
            for inputs, targets in self.valid_dataloader:
                # here only use 1 patch
                inputs, targets = inputs.float(), targets.float()
                if self.gpu == 1:
                    inputs, targets = inputs.cuda(), targets.cuda()
                preds_all += [self.model(inputs).cpu().numpy().squeeze()]
                targets_all += [targets.squeeze().tolist()]
            if self.gpu == 1:
                preds_all, targets_all = torch.tensor(preds_all).cuda(), torch.tensor(targets_all).cuda()
            else:
                preds_all, targets_all = torch.tensor(preds_all), torch.tensor(targets_all)

            if self.metric == 'Standard':
                loss = self.criterion(preds_all, targets_all)
            else:
                loss = self.criterion(preds_all, obss_all, hits_all)
        return loss

    def save_checkpoint(self, loss):
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(self.model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, self.model_name, self.optimal_epoch))
