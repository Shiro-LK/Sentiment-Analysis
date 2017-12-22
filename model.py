# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 14:04:02 2017

@author: shiro
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Lambda
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.layers import Conv1D, Dropout, LSTM, MaxPooling1D

def create_CBOW(input_dim, max_length, output_dim=50, n_class=2):
    model = Sequential()
    model.add(Embedding(input_dim+1, output_dim, input_length=max_length))
    model.add(Lambda(lambda x : K.sum(x,axis=1), output_shape=(output_dim,)))
    
    #model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_class, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    return model

def create_simpleModel(input_dim, max_length, output_dim=50, n_class=2):
    model = Sequential()
    model.add(Embedding(input_dim+1, output_dim, input_length=max_length))
    #model.add(Lambda(lambda x : K.sum(x,axis=1), output_shape=(output_dim,)))
    
    model.add(Conv1D(64, 3, border_mode='same')) # (length, input_dim = output dim of the embedding)
   
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='elu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_class, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    return model
def create_lstm(input_dim, max_length, output_dim=10, n_class=2):
    model = Sequential()
    model.add(Embedding(input_dim+1, output_dim, input_length=max_length))
    #model.add(Flatten())
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(n_class, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    return model