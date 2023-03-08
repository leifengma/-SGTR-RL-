import numpy as np
import pandas as pd
import os
import typing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# from sklearn import preprocessing

train_size, val_size = 0.9, 0.05



def get_concat_data(data):
    total_data_frame=pd.concat(data,ignore_index=True)
    return total_data_frame



def preprocess(data_array: np.ndarray, train_size: float, val_size: float,mean: float,std: float,self_mean_std=True):
    """Splits data into train/val/test sets and normalizes the data.

    Args:
        data_array: ndarray of shape `(num_time_steps, num_routes)`
        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.

    Returns:
        `train_array`, `val_array`, `test_array`
    """

    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    # train_array = data_array[:num_train]
    # # mean, std = train_array.mean(axis=0), train_array.std(axis=0)
    # #
    # # train_array = (train_array - mean) / std
    # val_array = data_array[num_train : (num_train + num_val)]
    # test_array = data_array[(num_train + num_val) :]

    # 计算标准化时启用
    train_array = data_array[:num_train]
    if self_mean_std:
        mean,std = mean, std
    else:
        mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val) :] - mean) / std

    return train_array, val_array, test_array, mean, std

def get_lstm_data(data,sequence_len,feature_len,predict_point):
    predict_point=predict_point-1
    features=[]
    target=[]
    data_len=len(data)
    for i in range(0,data_len-sequence_len-predict_point):
        X=data.iloc[i:sequence_len+i][:].to_numpy()
        y=data.iloc[sequence_len+i+predict_point][0:feature_len].to_numpy()
        features.append(X)
        target.append(y)
    return np.array(features), np.array(target)


def create_batch_data(X,y,train=True,buffer_size=1000,batch_size=128):
    batch_data=tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
    if train:
        return batch_data.cache().shuffle(buffer_size).batch(batch_size)
    else:
        return batch_data.batch(batch_size)


def get_data(divide:bool,source_data,feature_len,mean,std,predict_point,self_mean_std=True):
    if divide:
        train_datas, val_datas, test_datas,train_labels, val_labels, test_labels=[],[],[],[],[],[]
        for pdframe in source_data:
            train_array, val_array, test_array, mean, std = preprocess(pdframe, train_size, val_size,mean,std,self_mean_std)
            train_dataset,train_label=get_lstm_data(train_array,10,feature_len,predict_point)
            val_dataset,val_label=get_lstm_data(val_array,10,feature_len,predict_point)
            test_dataset,test_label=get_lstm_data(test_array,10,feature_len,predict_point)
            train_datas.append(train_dataset)
            val_datas.append(val_dataset)
            test_datas.append(test_dataset)
            train_labels.append(train_label)
            val_labels.append(val_label)
            test_labels.append(test_label)
        train_datasets,train_label_sets=np.concatenate(train_datas, axis=0),np.concatenate(train_labels, axis=0)
        train_batch_dataset=create_batch_data(train_datasets,train_label_sets)
        val_datasets,val_label_sets=np.concatenate(val_datas, axis=0),np.concatenate(val_labels, axis=0)
        val_batch_dataset=create_batch_data(val_datasets,val_label_sets)
        test_datasets,test_label_sets=np.concatenate(test_datas, axis=0),np.concatenate(test_labels, axis=0)
        test_batch_dataset=create_batch_data(test_datasets,test_label_sets)
        return train_batch_dataset,val_batch_dataset,test_batch_dataset,train_datasets,train_label_sets,val_datasets,val_label_sets,test_datasets,test_label_sets,mean,std
    else:
        train_array, val_array, test_array, mean, std = preprocess(source_data, 0.99, 0.005, mean, std, self_mean_std)
        train_datasets,train_label_sets=get_lstm_data(train_array,10,feature_len)
        val_datasets,val_label_sets=get_lstm_data(val_array,10,feature_len)
        test_datasets,test_label_sets=get_lstm_data(test_array,10,feature_len)
        train_batch_dataset=create_batch_data(train_datasets,train_label_sets)
        val_batch_dataset=create_batch_data(val_datasets,val_label_sets)
        test_batch_dataset=create_batch_data(test_datasets,test_label_sets)
        return train_batch_dataset,val_batch_dataset,test_batch_dataset,train_datasets,train_label_sets,val_datasets,val_label_sets,test_datasets,test_label_sets,mean,std


def create_model(inputshape,outputshape):
    model=keras.Sequential([
        layers.LSTM(units=128,input_shape=inputshape,return_sequences=True),
        layers.Dropout(0.4),
        layers.LSTM(units=64,return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(units=64,return_sequences=True),
        layers.LSTM(units=32),
        layers.Dense(outputshape)
    ])
    #keras.utils.plot_model(model)
    model.compile(optimizer='sgd',loss='mse')
    return model