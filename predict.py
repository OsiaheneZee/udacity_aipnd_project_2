# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:58:05 2020

@author: Michael Darko Ahwireng
"""
import predict_funcs as pf
import json
import argparse
from workspace_utils import active_session



parser = argparse.ArgumentParser(description='Enter variables needed for predicting the name of a flower')
parser.add_argument('img_path',type=str, metavar='-', help='the path to the image to be classified')
parser.add_argument('checkpoint',type=str, metavar='-', help='the path to the saved model to be used for classification')
parser.add_argument('--topk',type=int, metavar='-', help='the number of top predictions to be predicted by the model')
parser.add_argument('--category_names',type=str, metavar='-', help='the jason file of names of all the categories the flower can fall in', default = 'cat_to_name.json')
parser.add_argument('--gpu', metavar='-', action='store_const', const='gpu', help='the type of device, must be either gpu or cpu')

args = parser.parse_args()



with active_session():
    img_path= args.img_path
    model_path = args.checkpoint
    category_names = args.category_names
    topk = args.topk
    device = args.gpu

    model = pf.load_model(model_path=model_path)[0]


    with open(category_names, 'r') as json_file:
        cat_to_name = json.load(json_file)

    probs, labels = pf.predict(img_path=img_path, model=model, device=device, topk=topk)[0]
    probs_label_zip = list(zip(probs,labels))
    probs_label_zip.sort(reverse=True)

    print('The predicted top {} are :'.format(topk))

    for order,prob_lab in enumerate(probs_label_zip,1):
        print('\n{}. {} with a probabilty of {}'.format(order, prob_lab[1], prob_lab[0] ))


