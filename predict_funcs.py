# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:38:55 2020

@author: Michael Darko Ahwireng
"""
import torch
from PIL import Image
from torchvision import transforms, models




def load_model(model_path):
    '''loads a saved model and returns the model and the optimizer. takes the 
    path to the saved the model as input and returns the loaded model and 
    optimizer in that order'''
    checkpoint = torch.load(model_path)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict = (checkpoint['state_dict'])
    
    
    return model
    




def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        takes the path to the image as input and returns a transformed image
    '''
    pil_img = Image.open(img_path)
    
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    
    return img_transforms(pil_img)



def predict(img_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model,
    takes the path to the image(img_path), model(model) and the number of classes to predict(topk),
    returns the probabilities that the predicted class is right and the predicted class
    '''   
    
    img = process_image(img_path)
    img = img.unsqueeze_(0)
    img = img.float()
    
    #select device
    if device == 'gpu':
        device = torch.device('cuda')
        
    else:
        device = torch.device('cpu')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #send image to device
    img = img.to(device)
    model = model.to(device)
    
    with torch.no_grad():
        output = model.forward(img)
    probs = torch.exp(output).data
    probs, classes = probs.topk(topk)
    classes = classes.cpu()[0].tolist()
    probs = probs.cpu().tolist()[0]
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    # transfer index to label
    label = [idx_to_class[i] for i in classes]
    return probs, label