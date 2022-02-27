'''
utils.py
Contians the util functions for EP project
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filename_s = '/scratch2/xzhou/ep/LA_crime/results/CNN/2019-01-29-17/block_' + str(121)# + '_train'
#filename_s = '/scratch2/xzhou/ep/LA_crime/results/MLP/2018-12-11-18/block_' + str(121)

def draw(filename_s, avg, single_file=False, draw_trend=False):
    if single_file:
        df = pd.read_csv(filename_s + '.csv')
        pred = df['prediction'].values
        truth = df['truth'].values
        fig, ax = plt.subplots()
        plt.plot(pred,  'r+-', label='prediction', markersize=5, linestyle='None')
        plt.plot(truth, 'ko-', label='truth', markersize=0, linewidth=1)
        ax.set(xlabel='time (hour)', ylabel='# of crimes', title='result')
        ax.grid()
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

        plt.savefig(filename_s + '.png')
        plt.close()

        if draw_trend:
            df = pd.read_csv(filename_s + '.csv')
            pred = np.cumsum(df['prediction'].values)
            truth = np.cumsum(df['truth'].values)
            avg = np.cumsum(np.full(truth.shape, avg))
            fig, ax = plt.subplots()
            plt.plot(pred,  'r+-', label='prediction', markersize=5, linestyle='None')
            plt.plot(avg,  'b+-', label='average', markersize=5, linestyle='None')
            plt.plot(truth, 'ko-', label='truth', markersize=0, linewidth=1)
            ax.set(xlabel='time (hour)', ylabel='total # of crimes', title='result')
            ax.grid()
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

            plt.savefig(filename_s + '_cumsum.png')
            plt.close()

        print('figure drawn')

    else:
        '''
        df = pd.read_csv('../data/Block_Columns.csv')
        non_zero_blocks = set(df.columns.values)
        squared_error = 0

        for block in range(256):
            if str(block) in non_zero_blocks:
                filename_s = '../results/MLP/block_' + str(block)
                df = pd.read_csv(filename_s + '.csv')
                pred = df['prediction'].values
                truth = df['truth'].values

                fig, ax = plt.subplots()
                plt.plot(pred,  'r+-', label='prediction', markersize=5, linestyle='None')
                plt.plot(truth, 'ko-', label='truth', markersize=0, linewidth=1)
                ax.set(xlabel='time (hour)', ylabel='# of crimes', title='result')
                ax.grid()
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

                plt.savefig(filename_s + '.png')
                plt.close()

                if(sum(pred) > 0):
                    print(pred, block)
                    '''

#draw(filename_s, True)
