'''
data.py
Contians the functions processing data for EP project
'''
import pandas as pd
import numpy as np
import tensorflow

def process_data(block_num, csvfilename, categorical=True):
    # train and test data
    df = pd.read_csv(csvfilename)
    #df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    #df.Weekday = df.Date.dt.weekday
    df = df.fillna(value=0)
    #2401*168*112 => 2401*168*256 => 2401*16*16*168
    df2 = pd.DataFrame()
    for i in range(256):
        df2[str(i)] = np.zeros(4416,dtype=int)
    for i in range(256):
        if str(i) in df:
            df2[str(i)] = df[str(i)]
    df = df2
    # make block_num the Y value, then delete it and the date column
    Y = df[str(block_num)]
    #del df['Date']
    #del df['Training_Split'] # don't need this right now
    signals = df.values
    '''
    temp = np.zeros((len(signals), 16, 16))
    for i in range(len(signals)):
        temp[i] = signals[i].reshape(16, 16)
    #signals = signals.transpose()
    signals = temp
    '''
    #print(signals.shape)
    label = Y.values

    # generate data and label for training and testing
    train_start = 0
    train_end = 3850
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

        train_y_piece = label[label_pointer+predict_hours] #here delayed 24 hours
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

        test_y_piece = label[label_pointer+predict_hours]
        test_y.append(test_y_piece)

        label_pointer += 1
        pointer += 1
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    #print(test_x.shape, train_x.shape)
    average = np.mean(train_y)

    if categorical:
        num_categories = int(max(train_x.max(), test_x.max()))
        #print(num_categories)
        train_y = tensorflow.keras.utils.to_categorical(train_y, num_classes=num_categories)
        # don't need this line because Keras never sees the test vector
        test_y = tensorflow.keras.utils.to_categorical(test_y, num_classes=num_categories)

    return train_x, train_y, test_x, test_y, average
