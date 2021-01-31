import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
import re
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import os
import pickle
from model import *
from utils import *
import csv
import random
import nltk
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


imdb = pd.read_csv('.../IMDB Dataset.csv')
imdb.loc[imdb['sentiment']=='positive', 'sentiment'] = 0
imdb.loc[imdb['sentiment']=='negative', 'sentiment'] = 1

X_train, X_test, y_train, y_test = train_test_split(imdb['review'], imdb['sentiment'], test_size=0.2)

train_data = preprocess(X_train, max_length=384, device=device)
test_data = preprocess(X_test, max_length=384, device=device)
y_train = torch.tensor(y_train.values.astype(int), device=device)
y_test = torch.tensor(y_test.values.astype(int), device=device)


train_dataset = Data.TensorDataset(train_data, y_train)
test_dataset = Data.TensorDataset(test_data, y_test)

train_dataloader = Data.DataLoader(train_dataset, batch_size=32)
test_dataloader = Data.DataLoader(test_dataset, batch_size=32)


nasmi = NASMI(config, 2, device, training=True).cuda()
optimizer = optim.Adam(nasmi.parameters(), lr=0.001, eps=1e-8)
CE = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)


save_weight_path = '.../alpha_weight/cola'
save_model_path = '.../model_weight/cola'
if not os.path.isdir(save_weight_path):
    os.mkdir(save_weight_path)
if not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)
    
epochs = 10
stopping_round = 0
test_accuracy_list = []
for epoch in tqdm(range(epochs)):
    train_predict_result = []
    CE_mean_loss = []
    discrete_mean_loss = []
    first_mean_loss = []
    second_mean_loss = []
    nasmi.train()
    for X_train_batch, y_train_batch in train_dataloader:
        output, total_discrete_z, discrete_loss = nasmi(X_train_batch)
        softmax = nn.Softmax(dim=1)
        train_pred_prob = softmax(output)
        train_pred_prob = train_pred_prob.cpu().detach().numpy()[:,1]
        for prob in train_pred_prob:
            train_predict_result.append(prob)

        CE_loss = CE(output,torch.tensor(y_train_batch, dtype=torch.long).cuda())
        mean_layer_loss, first_loss, second_loss = Mutual_Information(total_discrete_z, alpha=1)
        total_loss = 0.5*(CE_loss) + 0.5(discrete_loss + mean_layer_loss)
        
        CE_mean_loss.append(CE_loss.cpu().detach().numpy())
        discrete_mean_loss.append(discrete_loss.cpu().detach().numpy())
        first_mean_loss.append(first_loss.cpu().detach().numpy())
        second_mean_loss.append(second_loss.cpu().detach().numpy())
        
        total_loss.backward()
        nn.utils.clip_grad_norm_(nasmi.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
    
    test_predict_result = []
    with torch.no_grad():
        nasmi.eval()
        for X_test_batch, y_test_batch in test_dataloader:
            output, _, _ = nasmi(X_test_batch)
            softmax = nn.Softmax(dim=1)
            test_pred_prob = softmax(output)
            test_pred_prob = test_pred_prob.cpu().detach().numpy()[:,1]
            for prob in test_pred_prob:
                test_predict_result.append(prob)
                
    structure_path = os.path.join(save_weight_path, 'NAS_structure{}.pkl'.format(epoch))

    CE_mean_loss = np.mean(CE_mean_loss)
    discrete_mean_loss = np.mean(discrete_mean_loss)
    first_mean_loss = np.mean(first_mean_loss)
    second_mean_loss = np.mean(second_mean_loss)
    
    train_predict_result = np.where(np.array(train_predict_result)>0.5,1,0)
    train_accuracy = accuracy_score(y_train.cpu(), train_predict_result)
    train_micro_f1 = f1_score(y_train.cpu(), train_predict_result, average='binary')
    train_macro_f1 = f1_score(y_train.cpu(), train_predict_result, average='binary')
    
    test_predict_result = np.where(np.array(test_predict_result)>0.5,1,0)
    test_accuracy = accuracy_score(y_test.cpu(), test_predict_result)
    test_micro_f1 = f1_score(y_test.cpu(), test_predict_result, average='binary')
    test_macro_f1 = f1_score(y_test.cpu(), test_predict_result, average='binary')
    test_accuracy_list.append(test_accuracy)
    
    model_weight_path = os.path.join(save_model_path, 'nasmi{}.pkl'.format(epoch))
    torch.save(nasmi.state_dict(), model_weight_path)
    
    print('Epochs:{},CE Loss:{:5f} Discrete Loss:{:5f} First Loss:{:5f} Second Loss:{:5f} train accuracy:{:5f} test accuracy:{:5f} train micro_f1:{:5f} train macro_f1:{:5f} test micro_f1:{:5f} test macro_f1:{:5f}'.format(
        epoch,CE_mean_loss,discrete_mean_loss,first_mean_loss, second_mean_loss, train_accuracy,test_accuracy, train_micro_f1, train_macro_f1, test_micro_f1, test_macro_f1))