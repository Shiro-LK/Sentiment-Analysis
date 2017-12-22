# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:50:50 2017

@author: shiro
"""

import preprocess
from preprocess import convert_data, vocab, create_data, fusion_data, loadTest, loadTexts, clean_str
from model import create_CBOW, create_simpleModel, create_lstm
import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
from keras.callbacks import ModelCheckpoint

from keras.models import load_model
import random
from sklearn.cross_validation import train_test_split

def max_len(x1, x2, x3):
    '''
        compute the maximum length of a sequence in the different dataset (x1, x2, x3)
    '''
    val = 1
    for x in x1:
        val = max([val, len(x)])
    for x in x2:
        val = max([val, len(x)])
    for x in x3:
        val = max([val, len(x)])
    return val
    
"""
    get a generator which choose image randomly
"""
def generator_shuffle(features, labels, num_classes, batch_size, dtype=np.int, input_shape=68):
     # Create empty arrays to contain batch of features and labels#
    '''
        features : ndarray
        labels : ndarray
        num_classes : int
        batch size : int
    '''
    while True:
        batch_features = np.ndarray(shape=(batch_size, input_shape), dtype=dtype)
        batch_labels =  np.ndarray(shape=(batch_size,  num_classes), dtype=dtype)

        index= np.random.randint(features.shape[0]-1, size=batch_size)
        #print(index)
        batch_features[:] = features[index]
        batch_labels[:] = labels[index]
        yield batch_features, batch_labels

"""
    Create simple generator
"""
def generator(features, labels, num_classes, batch_size, dtype=np.int, input_shape=68):
     # Create empty arrays to contain batch of features and labels#
    while True:
          for cpt in range(0, int(len(features)/batch_size)):
            #print('gen')
            batch_features = np.ndarray(shape=(batch_size, input_shape), dtype=dtype)
            batch_labels =  np.ndarray(shape=(batch_size, num_classes), dtype=dtype)
            for i in range(0, batch_size):
                index = cpt*batch_size + i
                batch_features[i] = features[index]
                batch_labels[i] = labels[index]
            yield batch_features, batch_labels

def preprocess_pad(X_tr, Y_tr, X_val, Y_val, max_length) :
    x_tr = pad_sequences(np.asarray(X_tr), maxlen=max_length, padding='post')
    y_tr = np.asarray(Y_tr)
    y_val = np.asarray(Y_val)
    x_val = pad_sequences(np.asarray(X_val), maxlen=max_length, padding='post')
    return x_tr, y_tr, x_val, y_val
    
def train_model(x_tr, y_tr, x_val, y_val, max_length, size_voc, output_dim=100, batch_size = 50):
    # preprocess data
    x_tr, y_tr, x_val, y_val = preprocess_pad(x_tr, y_tr, x_val, y_val, max_length)
    print(x_tr.shape)
    print(y_tr.shape)
    print(x_val.shape)
    print(y_val.shape)
    model = create_lstm(input_dim=size_voc, max_length=max_length, output_dim=output_dim, n_class=2 )
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-5, nesterov=True)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    generator_train = generator_shuffle(x_tr, y_tr, 2 , batch_size=batch_size, input_shape=(max_length))
    generator_valid = generator(x_val, y_val, 2, batch_size=batch_size, input_shape=(max_length))
    
    step_train = int(len(x_tr)/batch_size)-1
    step_val = int(len(x_val)/batch_size)-1
    print('shape:', len(x_tr))
    print('shape:', len(x_val))
    print('step train :' , step_train)
    print('step test :' , step_val)
    
     # callback
    callback_tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, 
                                                       batch_size=32, write_graph=True, write_grads=False, 
                                                       write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
                                                       embeddings_metadata=None)
    checkpoints = ModelCheckpoint('CBOW_keras_model.hdf5', verbose=1, save_best_only=True, period=1) # -{epoch:02d}
    callbacks_list = [callback_tensorboard, checkpoints]
    
    # train 
    model.fit_generator(generator_train,
          steps_per_epoch=step_train,
          epochs=10,
          verbose=1,
          validation_data=generator_valid,
          validation_steps=step_val,
          callbacks=callbacks_list)
    return model
    
def main():
    #-- Load data --#

    imdb_pos = 'imdb/imdb.pos'
    imdb_neg = 'imdb/imdb.neg'
    rt_pos = 'imdb/rt_critics.pos'
    rt_neg = 'imdb/rt_critics.neg'
    data_imdb, etq_imdb = create_data(imdb_pos, imdb_neg)
    data_rt, etq_rt = create_data(rt_pos, rt_neg)
    
    # train data and test
    data_tr, etq_train = fusion_data(data_imdb, data_rt, etq_imdb, etq_rt)
    test_rt, label_rt = loadTest('imdb/rt_critics.test')
    
    # list vocabulaire
    list_vocab = vocab(data_tr+test_rt)
    print(len(list_vocab))
    size_vocab = len(list(list_vocab.keys()))
    
    # convert word to number for neural network
    data_train = convert_data(data_tr, list_vocab)
    data_test = convert_data(test_rt, list_vocab)
    
    # split train into train and valid data
    X_train, X_val, y_train, y_val = train_test_split(data_train, etq_train, test_size=0.2)
    
    # determine sequence max
    sequence_max = max_len(X_train, X_val, data_test)
    print('sequence max :', sequence_max)
    label_rt = np.asarray(label_rt)
    data_test = pad_sequences(np.asarray(data_test), maxlen=sequence_max, padding='post')
    #model = train_model(X_train,y_train, X_val, y_val, sequence_max, size_vocab, output_dim=300, batch_size = 50)
    model = load_model('CBOW_keras_model.hdf5')
    print(model.evaluate(data_test, label_rt))
    
if __name__ == "__main__":
    main()