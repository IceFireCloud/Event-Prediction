#networks.py
# constructs the attention-based CNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os, sys
import json

from torch.utils.data import Dataset, DataLoader
from dataloader import ABCNN_Data
from sklearn.metrics import accuracy_score, classification_report

class ABCNN_1D(nn.Module):

    def __init__(self, config):
        super(ABCNN_1D, self).__init__()
        
        # 1 input image channel, output channels, 3 convolution
        self.conv1 = nn.Conv1d(config['in_channel'], config['num_filters'], kernel_size=3)
        self.conv2 = nn.Conv1d(config['num_filters'], 2*config['num_filters'], kernel_size=3)
        self.fc1 = nn.Linear(2*config['num_filters'] * 27, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config['max_crime'])
        self.mp = nn.MaxPool1d(3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def  num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
class ABCNN_2D(nn.Module):

    def __init__(self):
        return

class Network():
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = json.loads(f.read())['ABCNN_1D']
        
        train_data = ABCNN_Data(datapath=config['datapath'], stage='train', block_num=196,dim=1)
        test_data = ABCNN_Data(datapath=config['datapath'], stage='train', block_num=196,dim=1)
        self.train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        
        config['max_crime'] = train_data.max
        self.model = ABCNN_1D(config).cuda()
        print(self.model)
        
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        # self.criterion = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss().cuda()
        
        self.config = config
        
    def train(self):
        self.model.train(True)
        for epoch in range(self.config['epochs']):
            self.optimizer.zero_grad()
            
            for inputs, labels in (self.train_dataloader):
                inputs = inputs.cuda().float()
                labels = labels.cuda()
                
                preds = self.model(inputs)
                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            
    def test(self):
        self.model.train(False)
        with torch.no_grad():
            preds_all = []
            labels_all = []
            for inputs, labels in (self.test_dataloader):
                inputs = inputs.float().cuda()
                labels = labels.cuda()
                
                preds = self.model(inputs).cpu()
                preds = [np.argmax(p) for p in preds]
                preds_all += preds
                labels_all += labels.cpu()
            
        print(np.array(preds_all))
        print(preds_all.shape)
        print('average accuracy:', accuracy_score([0]*len(labels_all), labels_all))
        print('accuracy:', accuracy_score(preds_all, labels_all))
        print('report:', classification_report(y_true=labels_all, y_pred=preds_all, labels=list(range(self.config['max_crime'])), zero_division=0))
    
        
if __name__ == '__main__':
    print('networks')
    net = Network('./config.json')
    net.train()
    net.test()