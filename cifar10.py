import os
import sys

import cPickle as pickle
import numpy as np

DIR_BINARIES='/home/ignacio/Downloads/cifar-10-batches-py/'

def unpickle(filename):
    f = open(filename, 'rb')
    dic = pickle.load(f)
    f.close()
    return dic

class CIFAR10:
    def __init__(self, batch_size=100):
        # Training set
        train_data_list = []
        self.train_labels = []
        for bi in range(1,6):
            d = unpickle(DIR_BINARIES+'data_batch_'+str(bi))
            train_data_list.append(d['data'])
            self.train_labels += d['labels']
        self.train_data = np.concatenate(train_data_list, axis=0)

        # Test set
        d = unpickle(DIR_BINARIES+'test_batch')
        self.test_data = d['data']
        self.test_labels = d['labels']

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

if __name__=='__main__':
    cifar10 = CIFAR10(batch_size=1000)
    while cifar10.getEpoch()<2:
        batch, batch_idx = cifar10.nextBatch()
        print batch_idx, cifar10.n_batches, cifar10.getEpoch()
    batches = cifar10.getTestSet(asBatches=True)
    print len(batches)
    print batches[0][0].shape
