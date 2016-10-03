import os
import sys

import cPickle as pickle
import numpy as np

#DIR_BINARIES='/home/ignacio/Downloads/cifar-10-batches-py/'
DIR_BINARIES='/home/shared/cifar-10-batches-py/'

def unpickle(filename):
    f = open(filename, 'rb')
    dic = pickle.load(f)
    f.close()
    return dic

def batch_to_bc01(batch):
    ''' Converts CIFAR sample to bc01 tensor'''
    return batch.reshape([-1, 3, 32, 32])

def labels_to_one_hot(labels):
    ''' Converts list of integers to numpy 2D array with one-hot encoding'''
    labels = np.array(labels)
    N = len(labels)
    one_hot_labels = np.zeros([N, 10], dtype=int)
    one_hot_labels[np.arange(N), labels] = 1
    return one_hot_labels

class CIFAR10:
    def __init__(self, batch_size=100, validation_proportion=0.1):
        # Training set
        train_data_list = []
        self.train_labels = []
        for bi in range(1,6):
            d = unpickle(DIR_BINARIES+'data_batch_'+str(bi))
            train_data_list.append(d['data'])
            self.train_labels += d['labels']
        self.train_data = np.concatenate(train_data_list, axis=0).astype(np.float32)

        # Validation set
        assert validation_proportion > 0. and validation_proportion < 1.
        n_val = int(len(self.train_labels)*validation_proportion)
        self.validation_data = self.train_data[-n_val:]
        self.validation_labels = self.train_labels[-n_val:]
        self.train_data = self.train_data[:-n_val]
        self.train_labels = self.train_labels[:-n_val]
        
        # Test set
        d = unpickle(DIR_BINARIES+'test_batch')
        self.test_data = d['data'].astype(np.float32)
        self.test_labels = d['labels']

        # Normalize data
        mean = self.train_data.mean(axis=0)
        std = self.train_data.std(axis=0)
        self.train_data = (self.train_data-mean)/std
        self.test_data = (self.test_data-mean)/std

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

        # bc01 + one-hot
        batch_data = batch_to_bc01(batch_data)
        batch_labels = labels_to_one_hot(batch_labels)
        
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

                # bc01 + one-hot
                batch_data = batch_to_bc01(batch_data)
                batch_labels = labels_to_one_hot(batch_labels)
        
                batches.append((batch_data, batch_labels))
            return batches
        else:
            return (batch_to_bc01(self.test_data),
                    labels_to_one_hot(self.test_labels))

    def getValidationSet(self, asBatches=False):
        if asBatches:
            batches = []
            for i in range(len(self.validation_labels)//self.batch_size):
                start_idx = i*self.batch_size
                end_idx = start_idx + self.batch_size 
                batch_data = self.validation_data[start_idx:end_idx]
                batch_labels = self.validation_labels[start_idx:end_idx]

                # bc01 + one-hot
                batch_data = batch_to_bc01(batch_data)
                batch_labels = labels_to_one_hot(batch_labels)
        
                batches.append((batch_data, batch_labels))
            return batches
        else:
            return (batch_to_bc01(self.validation_data),
                    labels_to_one_hot(self.validation_labels))
        
if __name__=='__main__':
    cifar10 = CIFAR10(batch_size=1000)
    while cifar10.getEpoch()<2:
        batch, batch_idx = cifar10.nextBatch()
        print batch_idx, cifar10.n_batches, cifar10.getEpoch()
    batches = cifar10.getTestSet(asBatches=True)
    print len(batches)
    batches = cifar10.getValidationSet(asBatches=True)
    print len(batches)

    print batch[0][0][0]
