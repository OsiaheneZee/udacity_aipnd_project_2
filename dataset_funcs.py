# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:55:57 2020

@author: Michael Darko Ahwireng
"""

import torch
from torchvision import datasets, transforms




#transforms
train_transforms = transforms.Compose([
                                    transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                    ])



valid_transforms = transforms.Compose([
                                    transforms.Resize(225),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                    ])



test_transforms = transforms.Compose([
                                    transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                    ])



# datasets with ImageFolder

def create_datasets(train_dir, valid_dir, test_dir):
    '''this takes training data directory(train_dir), validation data directory(valid_dir) and 
    test_data directory(test_dir) and returns training, validation and test data in this same order'''
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    return train_data, valid_data, test_data

# dataloaders
    
def create_dataloaders(train_data, valid_data, test_data, batch_sizes = [64, 32, 32], shuffles = [True, False, False]):
    '''this takes training data(train_data), validation data(valid_data), 
    test_data(test_data), a list of batch sizes(batch_sizes) for trainloader=index 0, 
    vloader=index 1,and testloader=index2, and a list of shuffle values(shuffles) 
    which are boolean in the same order as batch sizes and returns training , validation and 
    test data dataloaders in this same order'''
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_sizes[0], shuffle=shuffles[0])
    vloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_sizes[1], shuffle=shuffles[1])
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_sizes[2], shuffle=shuffles[2])
    
    return trainloader, vloader, testloader