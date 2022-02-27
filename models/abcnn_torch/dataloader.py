# dataloader.py
# prepare the data for the network

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ABCNN_Data(Dataset):
    # dataset for AB CNN
    
    def __init__ (self, datapath, stage, block_num=None, dim=1):
        self.stage = stage
        df = pd.read_csv(datapath)
        df = df.fillna(value=0)
        df2 = pd.DataFrame()
        #2401*168*112 => 2401*168*256 => 2401*16*16*168
        #fill the non-listed regions with 0s
        for i in range(256):
            df2[str(i)] = np.zeros(4416,dtype=int)
        for i in range(256):
            if str(i) in df:
                df2[str(i)] = df[str(i)]
        df = df2
        if block_num:
            # make block_num the target value
            label = df[str(block_num)]
        else:
            label = df.values
        signals = df.values
        self.max = np.max(signals)
        if dim == 1:
            signals = signals
        elif dim == 2:
            temp = np.zeros((len(signals), 16, 16))
            for i in range(len(signals)):
                temp[i] = signals[i].reshape(16, 16)
            #signals = signals.transpose()
            signals = temp
            
        #print(signals.shape)
        label = label.values

        # generate data and label for training and testing
        # total = 4416 cases
        train_start = 0
        train_end = 3850
        input_len = 7*24
        predict_start= train_end + input_len
        predict_end = predict_start + 14*24
        predict_hours = 1*24

        train_x = []
        train_y = []
        pointer = train_start
        label_pointer = pointer + input_len - 1
        while label_pointer <= train_end:
            train_x_piece = signals[pointer:pointer+input_len]
            train_x.append(train_x_piece)

            train_y_piece = label[label_pointer+predict_hours] #here delayed 24 hours
            train_y.append(train_y_piece)

            pointer += 1
            label_pointer += 1

        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)

        test_x = []
        test_y = []
        # pointer
        label_pointer = predict_start
        pointer = label_pointer - input_len + 1
        while label_pointer <= predict_end:
            test_x_piece = signals[pointer:pointer + input_len]
            test_x.append(test_x_piece)

            test_y_piece = label[label_pointer+predict_hours]
            test_y.append(test_y_piece)

            label_pointer += 1
            pointer += 1
        self.test_x = np.array(test_x)
        self.test_y = np.array(test_y)
        # print(self.test_x.shape, self.train_x.shape)
        # print(self.test_y.shape, self.train_y.shape)
        self.average = np.mean(train_y)
        
    def __len__(self):
        if self.stage == 'train':
            return len(self.train_x)
        elif self.stage == 'test':
            return len(self.test_x)
        return 0
        
    def __getitem__(self, idx):
        if self.stage == 'train':
            return self.train_x[idx], self.train_y[idx]
        elif self.stage == 'test':
            return self.test_x[idx], self.test_y[idx]
        return 0
        

if __name__ == '__main__':
    print('dataloader')
    csvfilename = '/scratch/xzhou/ep/LA_crime/data/Block_Columns.csv'
    data = ABCNN_Data(datapath=csvfilename, stage='train', block_num=196,dim=1)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    for x,y in (dataloader):
        print(x.shape, y)
        break