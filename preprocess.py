# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:35:33 2017

@author: shiro
"""

import re
import numpy as np



def clean_str(string, tolower=True):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", " ", string)
    if tolower:
        string = string.lower()
    return string.strip()


def loadTexts(filename, limit=-1):
    f = open(filename, encoding="utf8")
    dataset=[]
    line =  f.readline()
    cpt=1
    skip=0
    while line :
        cleanline = clean_str(f.readline()).split()
        if cleanline: 
            dataset.append(cleanline)
        else: 
            line = f.readline()
            skip+=1
            continue
        if limit > 0 and cpt >= limit: 
            break
        line = f.readline()
        cpt+=1        
        
    f.close()
    print("Load ", cpt, " lines from ", filename , " / ", skip ," lines discarded")
    return dataset

def loadTest(filename, limit=-1):
    f = open(filename, encoding="utf8")
    dataset=[]
    label = []
    s = [line.strip() for line in f]
    for i in range(s.count('')):
        s.remove('')
    print(s.count('-1'))
    print(s.count('+1'))

    cpt=1
    skip=0
    for i, phrase in enumerate(s):
        if i % 2 == 0:
            cleanline = clean_str(phrase).split()
            if cleanline: 
                dataset.append(cleanline)
        else:
            if phrase == '-1':
                label.append(np.asarray([1, 0]))
            elif phrase == '+1':
                label.append(np.asarray([0, 1]))
            else:
                print('Erreur')
                break
    f.close()
    print("Load ", cpt, " lines from ", filename , " / ", skip ," lines discarded")
    return dataset, label
    
def create_data(critique_pos, critique_neg, binarize=False):
    '''
        pos , label = 1
        neg , label = 0
    '''
    data1 = loadTexts(critique_pos)
    data2 = loadTexts(critique_neg)
    etq1 = [np.asarray([0, 1]) for i in range(0, len(data1))]#np.ones((len(data1),), dtype=np.int)#[1 for i in range(0, len(data1))]
    etq2 = [np.asarray([1, 0]) for i in range(0, len(data2))]#np.zeros((len(data2)), dtype=np.int)#[0 for i in range(0, len(data2))]
    #-- --#
    data = data1 + data2
    #etq = np.append(etq1, etq2)
    etq = etq1 + etq2

    return data,  etq

def fusion_data(data1, data2, etq1, etq2):
    '''
        to fusion imdb and rt data
    '''
    etq = etq1 + etq2
    data = data1 + data2
    print('Nombre de données : ', len(data))
    return data, etq
    
def vocab(data):
    '''
        get vocabulary, and each word is associated to a number (between 1 and N)
    '''
    dic = {}
    it = 1
    for crit in data:
        for word in crit:
            if word not in dic:
                dic[word]=it
                it +=1
                
    return dic  

def convert_data(data, voc):
    '''
        convert word to number 
    '''
    newdata = []
    for cpt, crit in enumerate(data):
        temp = []
        #print('iteration : ', cpt)
        for word in crit:
            temp.append(voc[word])
        newdata.append(np.asarray(temp)) 
    return newdata