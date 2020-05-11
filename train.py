# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:28:07 2020

@author: Michael Darko Ahwireng
"""
import dataset_funcs as ds
import train_funcs as tf
import argparse
from workspace_utils import active_session




parser = argparse.ArgumentParser(description='Enter variables needed for the training of model')
parser.add_argument('data_directory',type=str, metavar='-', help='the path to the data to be used for the training')
parser.add_argument('--arch',type=str, metavar='-', help='the type of architures, must be one of the following vgg16, densenet121 or alexnet', default='vgg13')
parser.add_argument('--gpu', metavar='-', action='store_const', const='gpu', help='the type of device, must be either gpu or cpu')
parser.add_argument('--dropout',type=float, metavar='-', help='the type of probabilty between 0 and 1', default=0.5 )
parser.add_argument('--epochs',type=int, metavar='-', help='the number of epochs for training the model', default=20)
parser.add_argument('--hidden_layer',type=int, metavar='-', help='the size of input sample for the first hidden layer', default=512)
parser.add_argument('--learning_rate',type=float, metavar='-', help='the learning rate for the optimizer', default=0.01)
parser.add_argument('--print_every',type=int, metavar='-', help='the intervals in number of steps for the printing of training and validation loss', default=5)
parser.add_argument('--save_dir',type=str, metavar='-', help='the name by which the model is to be saved', default='checkpoint.pth')

args = parser.parse_args()


with active_session():
    arch = args.arch
    device = args.gpu
    dropout = args.dropout
    epochs = args.epochs
    hidden_layer1 = args.hidden_layer
    learning_rate = args.learning_rate
    print_every = args.print_every
    file_name = args.save_dir
    data_dir = args.data_directory

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    model = tf.model_select(arch=arch, dropout=dropout, hidden_layer1=hidden_layer1)
    optimizer = tf.model_classifier_optimizer(learning_rate=learning_rate, model=model)
    data = ds.create_datasets(train_dir, valid_dir, test_dir)
    train_data = data[0]
    valid_data = data[1]
    test_data = data[2]
    shuffles = {'epochs':epochs, 'dropout':dropout, 'hidden_lyr1':hidden_layer1}
    loaders = ds.create_dataloaders(train_data, valid_data, test_data)
    trainloader = loaders[0]
    vloader = loaders[1]
    testloader = loaders[2]
    model_params_dict = {'arch':arch, 'epochs':epochs, 'dropout':dropout, 'hidden_lyr1':hidden_layer1, 'learning_rate':learning_rate}

    tf.train_model(epochs=epochs, print_every=print_every, trainloader=trainloader, vloader=vloader, optimizer=optimizer,                   model=model, train_data=train_data, device=device)
    tf.model_test(model=model, testloader=testloader)
    tf.save_model(file_name=file_name, model=model, optimizer=optimizer, model_params_dict=model_params_dict)