# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 02:10:45 2023

@author: owner
"""

class EarlyStopping:
    def __int__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose
        
    def __call__(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
            
        else:
            self._step = 0
            self._loss = loss
            
        return False