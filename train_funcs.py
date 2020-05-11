# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:27:19 2020

@author: Michael Darko Ahwireng
"""

import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict




def  model_select(arch, dropout, hidden_layer1):
    model_dict = {'alexnet' : 9216, "vgg16":25088, 'vgg13':25088, "densenet121" : 1024}
    if arch not in model_dict:
        print('Please input one of the following \n{}'.format(list(model_dict.keys())))
    else:
        if arch == 'vgg16':
            model = models.vgg16(pretrained=True)
            
        elif arch == 'alexnet':
            model = models.alexnet(pretrained=True)
            
        elif arch == 'densenet121':
            model = models.densenet121(pretrained=True)
        
        #freeze features
        for param in model.parameters():
            param.requires_grad = False
        input_size = model_dict[arch]
        #create classifier
        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('fc1', nn.Linear(input_size, hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_layer1, 500)),
            ('relu2', nn.ReLU()),
            ('dropout', nn.Dropout(dropout)),
            ('fc3', nn.Linear(500, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    
        model.classifier = classifier
        
        return model
    

          
def model_classifier_optimizer(learning_rate, model):
    
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    return optimizer

#train the model

def train_model(epochs, print_every, trainloader, vloader, optimizer, model, train_data, device='gpu'):
    

    steps = 0
    criterion = nn.NLLLoss()

    #select 
    if device == 'gpu':
        device = torch.device('cuda')
        
    else:
        device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model to divice
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            model.train()
            steps += 1

            # Move input and label tensors to the device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy = 0

            
                for v_inputs,v_labels in vloader:
                    optimizer.zero_grad()
                    v_inputs, v_labels = v_inputs.to(device) , v_labels.to(device)
                    model.to(device)

                    with torch.no_grad():    
                        outputs = model.forward(v_inputs)
                        vlost = criterion(outputs,v_labels)
                        ps = torch.exp(outputs).data
                        equality = (v_labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                vlost = vlost / len(vloader)
                accuracy = accuracy /len(vloader)

                    
            
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(vlost),
                       "Accuracy: {:.2f}".format(accuracy))


                running_loss = 0
                
    model.class_to_idx = train_data.class_to_idx
                

                
                
def model_test(model, testloader):
    model.eval()
    correct_pred = 0
    total_pred = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for inputs, labels in testloader:
        # Move input and label tensors to the device
        inputs, labels = inputs.to(device), labels.to(device)
        probs = model(inputs)
        max_probs, model_pred = torch.max(probs.data, 1)
        correct_pred += (model_pred == labels).sum().item()
        total_pred += inputs.size()[0]
        
    mod_perf = (correct_pred/total_pred)*100
    
    print('The model has an accuracy of {:.2f}'.format(mod_perf))
    
    
def save_model(file_name, model, optimizer, model_params_dict):
    '''saves the model and the optimizer. The inputs are the name to save 
    the model by(file_name), model(model), optimizer(optimizer) and a dictionary 
    having epochs(epochs), dropout(dropout) and input for the first hidden 
    layer(hidden_lyr1) and learning_rate as keys'''
    model.cpu()
    torch.save({'arch' :model_params_dict['arch'],
                'hidden_lyr1':model_params_dict['hidden_lyr1'],
                'epochs':model_params_dict['epochs'],
                'dropout':model_params_dict['dropout'],
                'learning_rate':model_params_dict['learning_rate'],
                'state_dict':model.state_dict(),
                'classifier' : model.classifier,
                'optimizer':optimizer.state_dict(),
                'class_to_idx':model.class_to_idx},
                file_name)

