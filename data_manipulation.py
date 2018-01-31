##############################################################################
##
## data_manipulation.py
##
## @author: Matthew Cline
## @version: 20180129
##
## Description: Handle all of the shuffling and splitting of training and
## validation data. Exports pickle objects of the image paths and one hot
## encoded labels for both the training and the validation data.
##
##############################################################################

import pandas as pd
import numpy as np
import pickle

def splitData(data, trainingSplit=0.7):
    training, test = np.split(data, [int(data.shape[0] * trainingSplit)])
    return training, test

def shuffleData(data):
    data = data.reindex(np.random.permutation(data.index))
    data = data.reset_index(drop=True)
    return data

def encode_onehot(data):
    labels = []
    for row in data:
        if row == 1:
            labels.append([1,0,0,0,0,0,0,0,0,0])
        elif row == 2:
            labels.append([0,1,0,0,0,0,0,0,0,0])
        elif row == 3:
            labels.append([0,0,1,0,0,0,0,0,0,0])
        elif row == 4:
            labels.append([0,0,0,1,0,0,0,0,0,0])
        elif row == 5:
            labels.append([0,0,0,0,1,0,0,0,0,0])
        elif row == 6:
            labels.append([0,0,0,0,0,1,0,0,0,0])
        elif row == 7:
            labels.append([0,0,0,0,0,0,1,0,0,0])
        elif row == 8:
            labels.append([0,0,0,0,0,0,0,1,0,0])
        elif row == 9:
            labels.append([0,0,0,0,0,0,0,0,1,0])
        elif row == 10:
            labels.append([0,0,0,0,0,0,0,0,0,1])
    return np.array(labels)

print("\n\nReading the image paths and labels from data.csv...")
data = pd.read_csv('data.csv')

print("\n\nShuffling the data...")
data = shuffleData(data)

print("\n\nSplitting the data...")
train, val = splitData(data)

print("\n\nSeparating images and labels...")
trainImages = np.array(train.iloc[:,0])
valImages = np.array(val.iloc[:,0])
trainLabels = encode_onehot(np.array(train.iloc[:,1]))
valLabels = encode_onehot(np.array(val.iloc[:,1]))

print("\n\nDumping the data to pickle files...\n\n")
pickle.dump(trainImages, open("trainImages.p", "wb"))
pickle.dump(trainLabels, open("trainLabels.p", "wb"))
pickle.dump(valImages, open("valImages.p", "wb"))
pickle.dump(valLabels, open("valLabels.p", "wb"))