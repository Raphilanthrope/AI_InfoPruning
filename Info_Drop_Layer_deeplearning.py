#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:41:14 2021

@author: larsen
"""

# Code from "Information Pruning: a Regularization Method Based on a Simple Interpretability Property of Neural Networks"

import torch

class CIFAR10CNN_Baseline(torch.nn.Module): # name in paper: N
    
    def __init__(self, softplus=1):
        super(CIFAR10CNN_Baseline, self).__init__()
        
        self.softplus = softplus
        
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1) # 32
        self.act1 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1) # 32
        self.act2 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1) # 16
        self.act3 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        self.conv4 = torch.nn.Conv2d(64, 64, 3, padding=1) # 16
        self.act4 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv4.weight, nonlinearity="relu")
        self.pool2 = torch.nn.MaxPool2d(2, 2) 
        self.conv5 = torch.nn.Conv2d(64, 128, 3, padding=1) # 8
        self.act5 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv5.weight, nonlinearity="relu")
        self.conv6 = torch.nn.Conv2d(128, 128, 3, padding=1) # 8
        self.act6 = torch.nn.ReLU() 
        torch.nn.init.kaiming_normal_(self.conv6.weight, nonlinearity="relu")
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.act5 = torch.nn.ReLU()
        self.flat = torch.nn.Flatten(1)
        self.linear1 = torch.nn.Linear(4*4*128, 128)#
        self.act7 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        self.linear2 = torch.nn.Linear(128, 10)
        
        """self.linear1_bis = torch.nn.Linear(4*4*128, 10)#
        self.act8 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1_bis.weight, nonlinearity="relu")
        self.linear2_bis = torch.nn.Linear(10, 10)
        self.act9 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear2_bis.weight, nonlinearity="relu")
        self.linear3_bis = torch.nn.Linear(10, 1)
        """
    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool2(x)
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.pool3(x)
        
        """x_2 = self.flat(x)
        x_2 = self.act8(self.linear1_bis(x_2))
        x_2 = self.act9(self.linear2_bis(x_2))
        x_2 = self.linear3_bis(x_2)
        
        if(self.softplus>0):
            x_1 = torch.nn.Softplus(beta=self.softplus)(self.flat(x)-x_2)
        else:
            x_1 = torch.nn.ReLU()(self.flat(x)-x_2)
        x_1 = self.act7(self.linear1(x_1))
        x_1 = self.linear2(x_1)
        
        return x_1, x_2, torch.max(self.flat(x), dim=1)[0]"""
    
        x_1 = self.flat(x)
        x_1 = self.act7(self.linear1(x_1))
        x_1 = self.linear2(x_1)
        return x_1
    
    

class CIFAR10CNN_info_pruning(torch.nn.Module): # name in the paper: N_{p}
    
    def __init__(self, softplus=1):
        super(CIFAR10CNN_info_pruning, self).__init__()
        
        self.softplus = softplus
        
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1) # 32
        self.act1 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1) # 32
        self.act2 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1) # 16
        self.act3 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        self.conv4 = torch.nn.Conv2d(64, 64, 3, padding=1) # 16
        self.act4 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv4.weight, nonlinearity="relu")
        self.pool2 = torch.nn.MaxPool2d(2, 2) 
        self.conv5 = torch.nn.Conv2d(64, 128, 3, padding=1) # 8
        self.act5 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv5.weight, nonlinearity="relu")
        self.conv6 = torch.nn.Conv2d(128, 128, 3, padding=1) # 8
        self.act6 = torch.nn.ReLU() 
        torch.nn.init.kaiming_normal_(self.conv6.weight, nonlinearity="relu")
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.act5 = torch.nn.ReLU()
        self.flat = torch.nn.Flatten(1)
        self.linear1 = torch.nn.Linear(4*4*128, 128)#
        self.act7 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        self.linear2 = torch.nn.Linear(128, 10)
        
        self.linear1_bis = torch.nn.Linear(4*4*128, 10)#
        self.act8 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1_bis.weight, nonlinearity="relu")
        self.linear2_bis = torch.nn.Linear(10, 10)
        self.act9 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear2_bis.weight, nonlinearity="relu")
        self.linear3_bis = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool2(x)
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.pool3(x)
        
        x_2 = self.flat(x)
        x_2 = self.act8(self.linear1_bis(x_2))
        x_2 = self.act9(self.linear2_bis(x_2))
        x_2 = self.linear3_bis(x_2)
        
        if(self.softplus>0):
            x_1 = torch.nn.Softplus(beta=self.softplus)(self.flat(x)-x_2)
        else:
            x_1 = torch.nn.ReLU()(self.flat(x)-x_2)
        x_1 = self.act7(self.linear1(x_1))
        x_1 = self.linear2(x_1)
        
        return x_1, x_2, torch.max(self.flat(x), dim=1)[0]


class CIFAR10CNNplus(torch.nn.Module): # name in the paper: N_{+}
    
    def __init__(self):
        super(CIFAR10CNNplus, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1) # 32
        self.act1 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1) # 32
        self.act2 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1) # 16
        self.act3 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        self.conv4 = torch.nn.Conv2d(64, 64, 3, padding=1) # 16
        self.act4 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv4.weight, nonlinearity="relu")
        self.pool2 = torch.nn.MaxPool2d(2, 2) 
        self.conv5 = torch.nn.Conv2d(64, 128, 3, padding=1) # 8
        self.act5 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv5.weight, nonlinearity="relu")
        self.conv6 = torch.nn.Conv2d(128, 128, 3, padding=1) # 8
        self.act6 = torch.nn.ReLU() 
        torch.nn.init.kaiming_normal_(self.conv6.weight, nonlinearity="relu")
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.act5 = torch.nn.ReLU()
        self.flat = torch.nn.Flatten(1)
        self.linear1 = torch.nn.Linear(4*4*128, 139)#
        self.act7 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        self.linear2 = torch.nn.Linear(139, 10)
    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool2(x)
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.pool3(x)
        x_1 = self.flat(x)
        x_1 = self.act7(self.linear1(x_1))
        x_1 = self.linear2(x_1)
        #x_1 = torch.nn.functional.softmax(x_1, dim=1)
        return x_1
        

class CIFAR10CNN_Baseline_Drop(torch.nn.Module): # name in paper: N_d
    
    def __init__(self, softplus=1):
        super(CIFAR10CNN_Baseline_Drop, self).__init__()
        
        self.softplus = softplus
        
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1) # 32
        self.act1 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1) # 32
        self.act2 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1) # 16
        self.act3 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        self.conv4 = torch.nn.Conv2d(64, 64, 3, padding=1) # 16
        self.act4 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv4.weight, nonlinearity="relu")
        self.pool2 = torch.nn.MaxPool2d(2, 2) 
        self.conv5 = torch.nn.Conv2d(64, 128, 3, padding=1) # 8
        self.act5 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv5.weight, nonlinearity="relu")
        self.conv6 = torch.nn.Conv2d(128, 128, 3, padding=1) # 8
        self.act6 = torch.nn.ReLU() 
        torch.nn.init.kaiming_normal_(self.conv6.weight, nonlinearity="relu")
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.act5 = torch.nn.ReLU()
        self.flat = torch.nn.Flatten(1)
        self.linear1 = torch.nn.Linear(4*4*128, 128)#
        self.act7 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        self.linear2 = torch.nn.Linear(128, 10)
        
        """self.linear1_bis = torch.nn.Linear(4*4*128, 10)#
        self.act8 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1_bis.weight, nonlinearity="relu")
        self.linear2_bis = torch.nn.Linear(10, 10)
        self.act9 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear2_bis.weight, nonlinearity="relu")
        self.linear3_bis = torch.nn.Linear(10, 1)
        """
    
    def forward(self, x, p=0.2):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)
        x = torch.nn.functional.dropout(x, p=p, training=self.training) 
        
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool2(x)
        x = torch.nn.functional.dropout(x, p=p, training=self.training) 
        
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.pool3(x)
        x = torch.nn.functional.dropout(x, p=p, training=self.training) 
        
        """x_2 = self.flat(x)
        x_2 = self.act8(self.linear1_bis(x_2))
        x_2 = self.act9(self.linear2_bis(x_2))
        x_2 = self.linear3_bis(x_2)
        
        if(self.softplus>0):
            x_1 = torch.nn.Softplus(beta=self.softplus)(self.flat(x)-x_2)
        else:
            x_1 = torch.nn.ReLU()(self.flat(x)-x_2)
        x_1 = self.act7(self.linear1(x_1))
        x_1 = self.linear2(x_1)
        
        return x_1, x_2, torch.max(self.flat(x), dim=1)[0]"""
    
        x_1 = self.flat(x)
        x_1 = self.act7(self.linear1(x_1))
        x_1 = torch.nn.functional.dropout(x_1, p=p, training=self.training) 
        x_1 = self.linear2(x_1)
        return x_1
    

class CIFAR10CNN_info_pruning_drop(torch.nn.Module): # name in the paper: N_{d+p}
    
    def __init__(self, softplus=1):
        super(CIFAR10CNN_info_pruning_drop, self).__init__()
        
        self.softplus = softplus
        
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1) # 32
        self.act1 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1) # 32
        self.act2 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1) # 16
        self.act3 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        self.conv4 = torch.nn.Conv2d(64, 64, 3, padding=1) # 16
        self.act4 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv4.weight, nonlinearity="relu")
        self.pool2 = torch.nn.MaxPool2d(2, 2) 
        self.conv5 = torch.nn.Conv2d(64, 124, 3, padding=1) # 8
        self.act5 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv5.weight, nonlinearity="relu")
        self.conv6 = torch.nn.Conv2d(124, 124, 3, padding=1) # 8
        self.act6 = torch.nn.ReLU() 
        torch.nn.init.kaiming_normal_(self.conv6.weight, nonlinearity="relu")
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.act5 = torch.nn.ReLU()
        self.flat = torch.nn.Flatten(1)
        self.linear1 = torch.nn.Linear(4*4*124, 124)#
        self.act7 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        self.linear2 = torch.nn.Linear(124, 10)
        
        self.linear1_bis = torch.nn.Linear(4*4*124, 10)#
        self.act8 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1_bis.weight, nonlinearity="relu")
        self.linear2_bis = torch.nn.Linear(10, 10)
        self.act9 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear2_bis.weight, nonlinearity="relu")
        self.linear3_bis = torch.nn.Linear(10, 1)
    
    def forward(self, x, p=0.2):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)
        x = torch.nn.functional.dropout(x, p=p, training=self.training)        

        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool2(x)        
        x = torch.nn.functional.dropout(x, p=p, training=self.training)        

        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.pool3(x)
        x = torch.nn.functional.dropout(x, p=p, training=self.training)        

        x_2 = self.flat(x)
        x_2 = self.act8(self.linear1_bis(x_2))
        x_2 = self.act9(self.linear2_bis(x_2))
        x_2 = self.linear3_bis(x_2)
        
        if(self.softplus>0):
            x_1 = torch.nn.Softplus(beta=self.softplus)(self.flat(x)-x_2)
        else:
            x_1 = torch.nn.ReLU()(self.flat(x)-x_2)
        x_1 = self.act7(self.linear1(x_1))        
        x_1 = torch.nn.functional.dropout(x_1, p=p, training=self.training)        
        x_1 = self.linear2(x_1)
        
        return x_1, x_2, torch.max(self.flat(x), dim=1)[0]
    
    
class CIFAR10CNN_2_info_pruning(torch.nn.Module): # name in the paper: N_{2p}
    
    def __init__(self, softplus=1):
        super(CIFAR10CNN_2_info_pruning, self).__init__()
        
        self.softplus = softplus
        
        self.flat = torch.nn.Flatten(1)
        
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1) # 32
        self.act1 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        self.conv2 = torch.nn.Conv2d(32, 28, 3, padding=1) # 32
        self.act2 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        
        self.conv1bis = torch.nn.Conv2d(28, 8, 3, padding=1) # 16
        self.act10 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv1bis.weight, nonlinearity="relu")
        self.pool1bis = torch.nn.MaxPool2d(2, 2) # 8
        self.conv2bis = torch.nn.Conv2d(8, 4, 3, padding=1) # 8
        self.act11 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv2bis.weight, nonlinearity="relu")
        self.pool2bis = torch.nn.MaxPool2d(2, 2) # 4
        self.linear1_tris = torch.nn.Linear(64, 10)
        self.act12 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1_tris.weight, nonlinearity="relu")
        self.linear2_tris = torch.nn.Linear(10, 1)
        
        self.conv3 = torch.nn.Conv2d(28, 64, 3, padding=1) # 16
        self.act3 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        self.conv4 = torch.nn.Conv2d(64, 64, 3, padding=1) # 16
        self.act4 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv4.weight, nonlinearity="relu")
        self.pool2 = torch.nn.MaxPool2d(2, 2) 
        self.conv5 = torch.nn.Conv2d(64, 128, 3, padding=1) # 8
        self.act5 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.conv5.weight, nonlinearity="relu")
        self.conv6 = torch.nn.Conv2d(128, 126, 3, padding=1) # 8
        self.act6 = torch.nn.ReLU() 
        torch.nn.init.kaiming_normal_(self.conv6.weight, nonlinearity="relu")
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        
        self.linear1 = torch.nn.Linear(4*4*126, 120)#
        self.act7 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1.weight, nonlinearity="relu")
        self.linear2 = torch.nn.Linear(120, 10)
        
        self.linear1_bis = torch.nn.Linear(4*4*126, 10)#
        self.act8 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear1_bis.weight, nonlinearity="relu")
        self.linear2_bis = torch.nn.Linear(10, 10)
        self.act9 = torch.nn.ReLU()
        torch.nn.init.kaiming_normal_(self.linear2_bis.weight, nonlinearity="relu")
        self.linear3_bis = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)
        
        x_0 = self.act10(self.conv1bis(x))
        x_0 = self.pool1bis(x_0)
        x_0 = self.act11(self.conv2bis(x_0))
        x_0 = self.pool2bis(x_0)
        x_0 = self.flat(x_0)
        x_0 = self.act12(self.linear1_tris(x_0))
        x_0 = self.linear2_tris(x_0)
        x_0 = torch.unsqueeze(x_0.unsqueeze(-1), -1)
        
        if(self.softplus>0):
            x_1 = torch.nn.Softplus(beta=self.softplus)(x-x_0)
        else:
            x_1 = torch.nn.ReLU()(x-x_0)
        
        a = ((x_0 - torch.max(self.flat(x), dim=1)[0])**2).mean()        
        
        x = self.act3(self.conv3(x_1))
        x = self.act4(self.conv4(x))
        x = self.pool2(x)
        x = self.act5(self.conv5(x))
        x = self.act6(self.conv6(x))
        x = self.pool3(x)
        
        x_2 = self.flat(x)
        x_2 = self.act8(self.linear1_bis(x_2))
        x_2 = self.act9(self.linear2_bis(x_2))
        x_2 = self.linear3_bis(x_2)
        
        if(self.softplus>0):
            x_1 = torch.nn.Softplus(beta=self.softplus)(self.flat(x)-x_2)
        else:
            x_1 = torch.nn.ReLU()(self.flat(x)-x_2)
            
        x_1 = self.act7(self.linear1(x_1))
        x_1 = self.linear2(x_1)
        
        a = a + ((x_2 - torch.max(self.flat(x), dim=1)[0])**2).mean()
        
        return x_1, a 
    


import tensorflow as tf
        
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


train_torch = torch.tensor(np.moveaxis(x_train/float(np.max(x_train)), -1, 1), dtype=torch.float32)
lab = torch.tensor(y_train, dtype=torch.int64)

test_torch = torch.tensor(np.moveaxis(x_test/float(np.max(x_train)), -1, 1), dtype=torch.float32)
test_lab = torch.tensor(y_test, dtype=torch.int64).squeeze()


num_val = 5000
val = train_torch[-num_val:]
lab_val = lab[-num_val:].squeeze()


trainloader = torch.utils.data.DataLoader(list(zip(train_torch[:-num_val], lab[:-num_val])), batch_size=64,  shuffle=True)

model = CIFAR10CNN_Baseline()
# torch.save(model.state_dict(), "")
# model.load_state_dict(torch.load(""))
print(sum(p.numel() for p in model.parameters()))

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 

total=0
success=0
update = 100

alpha = 0.1

model.train()

for epoch in range(500):  
    
    running_loss = 0.0
    running_confusion = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        
        labels = labels.squeeze()

        optimizer.zero_grad()

        # outputs, thresh, max_pruned = model(inputs) # uncomment iff the model has an IPM
        # outputs, info_reg = model(inputs) # uncomment iff the model has an IPM with two nets
	# outputs = model(inputs)  # uncomment iff the model does not have an IPM

        # loss = criterion(outputs, labels) + alpha*((thresh - max_pruned)**2).mean()  # uncomment iff the model has an IPM with one Net 
	# loss = criterion(outputs, labels) + alpha*info_reg  # uncomment iff the model has an IPM with two nets
	# loss = criterion(outputs, labels)  # uncomment iff the model does not have an IPM

        loss.backward()
        optimizer.step()        
        
        _,predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        success += (predicted==labels.data).sum()

        running_loss += loss.item()
        
        if(i%update==update-1):    
            print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / update))
            running_loss = 0.0
            
            acc = (100.0*success/total).item()
            print("acc: "+str(acc))
            
            total = 0
            success = 0
            

    # torch.save(model.state_dict(), "..."+str(epoch+1)+".pt")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
