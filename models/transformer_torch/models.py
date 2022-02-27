# network models for vision transformer
# Created: 6/16/2021
# Status: in progress

import sys

import torch.nn as nn

import torch

class _ViT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        t_layer = nn.TransformerEncoderLayer(d_model=config['emsize'], nhead=config['nhead'], dim_feedforward=config['dim_feedforward'], dropout=config['dropout'])
        self.t_encoder = nn.TransformerEncoder(encoder_layer=t_layer, num_layers=config['nlayers'])
        self.map = nn.Linear(256, config['emsize'])
        # self.embed3 = nn.Embedding(100, config['emsize']) #for now sequence is only 1, will change to 27 once this runs||||||||||||||
        self.t_decoder = nn.Linear(config['emsize'], config['out_dim'])
        # self.da = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(2, 0, 1)
        # b_size = x.shape[1]
        # x1 = x[:,:,:60].flatten().view(b_size, -1) #in here shape[0] is the batch size
        # x3 = self.embed3(x3)
        # print('Embedding, before T-encoder', x3.shape)
        # print('v1', self.t_decoder(x3).shape)
        # print('v2', self.t_decoder(x3.view(b_size, -1)).shape)

        # print(x.shape)
        x = self.map(x)
        # x = torch.stack((x1, x2))
        # seq_len = 2
        # x = x.view(seq_len, b_size, -1)
        x = self.t_encoder(x)
        # print(x.shape)
        # sys.exit()
        # x = x.view(b_size, -1)
        # print('before linear', x.shape)
        # print(self.t_decoder)
        x = self.t_decoder(x)
        # x = self.da(x)
        # print('last', x.shape)
        x = x[0, :]
        # print('last first', x.shape)


        return x
