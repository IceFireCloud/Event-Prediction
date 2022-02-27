# dataloader for vision transformer
# Created: 6/16/2021
# Status: in progress

import random
import glob
import os, sys

import numpy as np
import nibabel as nib
import pandas as pd

from torch.utils.data import Dataset
from utils import read_csv_cox, rescale

SCALE = 1 #rescale to 0~2.5


class ViT_Data(Dataset):
    def __init__(self, exp_idx, stage, ratio=(0.6, 0.2, 0.2), seed=1000, name='', fold=[]):
        random.seed(seed)
        
        # self.process_data(block_num=0)
        csvfilename='/scratch/xzhou/ep/LA_crime/data/Block_Columns.csv'
        self.process_data(csvfilename=csvfilename)

        self.stage = stage
        self.exp_idx = exp_idx
        return
        # print(self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape, self.average)
        # sys.exit()

        # self.data_list = glob.glob(data_dir + 'coregistered*nii*')

        # csvname = '~/mri-pet/metadata/data_processed/merged_dataframe_cox_noqc_pruned_final.csv'
        # csvname = os.path.expanduser(csvname)

        # print(len(tmp_f))
        l = self.x.shape[0]
        split1 = int(l*ratio[0])
        split2 = int(l*(ratio[0]+ratio[1]))
        idxs = list(range(len(fileIDs)))
        random.shuffle(idxs)
        if 'train' in stage:
            self.index_list = idxs[:split1]
        elif 'valid' in stage:
            self.index_list = idxs[split1:split2]
        elif 'test' in stage:
            self.index_list = idxs[split2:]
        elif 'all' in stage:
            self.index_list = idxs
        else:
            raise Exception('Unexpected Stage for Vit_Data!')
        # print(len(self.index_list))
        # sys.exit()
    
    def process_data(self, csvfilename, categorical=False, block_num=None):
        if block_num:
            'not ready for single block prediction yet'
            sys.exit()
        # train and test data
        df = pd.read_csv(csvfilename)
        #df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
        #df.Weekday = df.Date.dt.weekday
        df = df.fillna(value=0)
        #[batch size, sequence length, width, height] -- entire area predict
        # optional: 
        #the target would instead be [337] rather than [337*16*16]
        #2401*168*112 => 2401*168*256 => 2401*16*16*168
        df2 = pd.DataFrame()
        for i in range(256):
            df2[str(i)] = np.zeros(4416,dtype=int)
        for i in range(256):
            if str(i) in df:
                df2[str(i)] = df[str(i)]
        df = df2
        # make block_num the Y value, then delete it and the date column
        # Y = df[str(block_num)]
        #del df['Date']
        #del df['Training_Split'] # don't need this right now
        signals = df.values
        img = False
        if img:
            temp = np.zeros((len(signals), 16, 16))
        else:
            temp = np.zeros((len(signals), 256))
        for i in range(len(signals)):
            if img:
                temp[i] = signals[i].reshape(16, 16)
            else:
                temp[i] = signals[i].reshape(256)
        #signals = signals.transpose()
        signals = temp

        # print(signals.shape)
        # sys.exit()
        # label = Y.values

        # generate data and label for training and testing
        train_start = 0
        train_end = 2567
        predict_start= train_end + 7*24
        predict_end = predict_start + 14*24
        input_len = 7*24
        predict_hours = 1*24
        #single output

        train_x = []
        train_y = []
        pointer = train_start
        label_pointer = pointer + input_len - 1
        while label_pointer <= train_end:
            #train_x_piece = signals[pointer:pointer+input_len, :, :]
            train_x_piece = signals[pointer:pointer+input_len, :]
            train_x.append(train_x_piece.transpose())

            train_y_piece = signals[label_pointer+predict_hours] #here delayed 24 hours
            # train_y_piece = label[label_pointer+predict_hours] #here delayed 24 hours
            train_y.append(train_y_piece)

            pointer += 1
            label_pointer += 1

        train_x = np.array(train_x)
        train_y = np.array(train_y)

        test_x = []
        test_y = []
        # pointer
        label_pointer = predict_start
        pointer = label_pointer - input_len + 1
        while label_pointer <= predict_end:
            #test_x_piece = signals[pointer:pointer + input_len, :, :]
            test_x_piece = signals[pointer:pointer + input_len, :]
            test_x.append(test_x_piece.transpose())

            test_y_piece = signals[label_pointer+predict_hours]
            # test_y_piece = label[label_pointer+predict_hours]
            test_y.append(test_y_piece)

            label_pointer += 1
            pointer += 1
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        #print(test_x.shape, train_x.shape)
        average = np.mean(train_y)

        self.num_categories = int(signals.max())
        self.all = signals
        if categorical:
            num_categories = int(signals.max())
            #print(num_categories)
            train_y = tensorflow.keras.utils.to_categorical(train_y, num_classes=num_categories)
            # don't need this line because Keras never sees the test vector
            test_y = tensorflow.keras.utils.to_categorical(test_y, num_classes=num_categories)
        self.train_x, self.train_y, self.test_x, self.test_y, self.average = train_x, train_y, test_x, test_y, average
        # print('sum(train_x)')
        # print(self.train_x.sum(), self.train_y.sum(), self.test_x.sum(), self.test_y.sum(), self.average)
        # sys.exit()
        return train_x, train_y, test_x, test_y, average

    def __len__(self):
        if self.stage == 'train':
            return self.train_x.shape[0]
        else:
            return self.test_x.shape[0]
            
        return len(self.index_list)
        
    def __getitem__(self, idx):
        if self.stage == 'train':
            return self.train_x[idx], self.train_y[idx]
        else:
            return self.test_x[idx], self.test_y[idx]
            
            
        idx = self.index_list[idx]
        obs = self.time_obs[idx]
        hit = self.time_hit[idx]

        data = nib.load(self.data_list[idx]).get_fdata().astype(np.float32)
        data[data != data] = 0
        if SCALE:
            data = rescale(data, (0, 2.5))
            if 0:
                data = rescale(data, (0, 99))
                data = data.astype(np.int)
        data = np.expand_dims(data, axis=0)
        return data, obs, hit

    def get_sample_weights(self):
        num_classes = self.num_categories
        counts = [(self.all == i).sum() for i in range(num_classes)]
        count = self.all.shape[0]
        print(count)
        print(counts)
        print('not supportted!')
        sys.exit()
        weights = [count / counts[i] for i in self.all]
        class_weights = [count/c for c in counts]
        return weights, class_weights
