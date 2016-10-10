import os
import sys

import cPickle as pickle
import numpy as np
from sklearn.model_selection import train_test_split

DIR_BINARIES='/home/ignacio/Downloads/cifar-10-batches-py/'

def unpickle(filename):
    f = open(filename, 'rb')
    dic = pickle.load(f)
    f.close()
    return dic

def batch_to_bc01(batch):
    ''' Converts CIFAR sample to bc01 tensor'''
    return batch.reshape([-1, 3, 32, 32])

def batch_to_b01c(batch):
    ''' Converts CIFAR sample to b01c tensor'''
    return batch_to_bc01(batch).transpose(0,2,3,1)

def labels_to_one_hot(labels):
    ''' Converts list of integers to numpy 2D array with one-hot encoding'''
    N = len(labels)
    one_hot_labels = np.zeros([N, 10], dtype=int)
    one_hot_labels[np.arange(N), labels] = 1
    return one_hot_labels

class CIFAR10:
    def __init__(self, batch_size=100, validation_proportion=0.1, augment_data=False):
        # Training set
        train_data_list = []
        self.train_labels = []
        for bi in range(1,6):
            d = unpickle(DIR_BINARIES+'data_batch_'+str(bi))
            train_data_list.append(d['data'])
            self.train_labels += d['labels']
        self.train_labels = np.asarray(self.train_labels)
        self.train_data = np.concatenate(train_data_list, axis=0).astype(np.float32)
        
        # Validation set
        assert validation_proportion > 0. and validation_proportion < 1.
        self.train_data, self.validation_data, self.train_labels, self.validation_labels = train_test_split(
            self.train_data, self.train_labels, test_size=validation_proportion, random_state=1)
                
        # Test set
        d = unpickle(DIR_BINARIES+'test_batch')
        self.test_data = d['data'].astype(np.float32)
        self.test_labels = np.asarray(d['labels'])

        # Normalize data
        mean = self.train_data.mean(axis=0)
        std = self.train_data.std(axis=0)
        self.train_data = (self.train_data-mean)/std
        self.validation_data = (self.validation_data-mean)/std
        self.test_data = (self.test_data-mean)/std

        # Converting to b01c and one-hot encoding
        self.train_data = batch_to_b01c(self.train_data)
        self.validation_data = batch_to_b01c(self.validation_data)
        self.test_data = batch_to_b01c(self.test_data)
        self.train_labels = labels_to_one_hot(self.train_labels)
        self.validation_labels = labels_to_one_hot(self.validation_labels)
        self.test_labels = labels_to_one_hot(self.test_labels)

        # Augment training dataset (horizontal flipping)
        np.random.seed(seed=1)
        if augment_data:
            flipped_train_data = self.train_data[:, :, ::-1, :]
            self.train_data = np.concatenate([self.train_data, flipped_train_data],
                                             axis=0)
            self.train_labels = np.concatenate([self.train_labels, self.train_labels],
                                               axis=0)
            
            # shuffle training set
            new_idx = np.random.permutation(np.arange(len(self.train_labels)))
            self.train_data = self.train_data[new_idx]
            self.train_labels = self.train_labels[new_idx]
            
        # Batching & epochs
        self.batch_size = batch_size
        self.n_batches = len(self.train_labels)//self.batch_size
        self.current_batch = 0
        self.current_epoch = 0
        
    def nextBatch(self):
        ''' Returns a tuple with batch and batch index '''
        start_idx = self.current_batch*self.batch_size
        end_idx = start_idx + self.batch_size 
        batch_data = self.train_data[start_idx:end_idx]
        batch_labels = self.train_labels[start_idx:end_idx]
        batch_idx = self.current_batch
        
        # Update self.current_batch and self.current_epoch
        self.current_batch = (self.current_batch+1)%self.n_batches
        if self.current_batch != batch_idx+1:
            self.current_epoch += 1
            
        return ((batch_data, batch_labels), batch_idx)
    
    def getEpoch(self):
        return self.current_epoch

    # TODO: refactor getTestSet and getValidationSet to avoid code replication
    def getTestSet(self, asBatches=False):
        if asBatches:
            batches = []
            for i in range(len(self.test_labels)//self.batch_size):
                start_idx = i*self.batch_size
                end_idx = start_idx + self.batch_size 
                batch_data = self.test_data[start_idx:end_idx]
                batch_labels = self.test_labels[start_idx:end_idx]
        
                batches.append((batch_data, batch_labels))
            return batches
        else:
            return (self.test_data, self.test_labels)

    def getValidationSet(self, asBatches=False):
        if asBatches:
            batches = []
            for i in range(len(self.validation_labels)//self.batch_size):
                start_idx = i*self.batch_size
                end_idx = start_idx + self.batch_size 
                batch_data = self.validation_data[start_idx:end_idx]
                batch_labels = self.validation_labels[start_idx:end_idx]

                batches.append((batch_data, batch_labels))
            return batches
        else:
            return (self.validation_data, self.validation_labels)

    def reset(self):
        self.current_batch = 0
        self.current_epoch = 0
        
if __name__=='__main__':
    cifar10 = CIFAR10(batch_size=1000)
    while cifar10.getEpoch()<2:
        batch, batch_idx = cifar10.nextBatch()
        print batch_idx, cifar10.n_batches, cifar10.getEpoch()
    batches = cifar10.getTestSet(asBatches=True)
    print len(batches)
    data, labels = cifar10.getValidationSet()
    print labels.sum(axis=0)
    data, labels = cifar10.getTestSet()
    print labels.sum(axis=0)
