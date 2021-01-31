import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import os
import json
from nltk.tokenize import WordPunctTokenizer
import spacy
import re
from tqdm import tqdm
import logging
import pickle
import math


class ResidualConnect(nn.Module):
    def __init__(self, config):
        super(ResidualConnect, self).__init__()
        self.hidden_size = config['hidden_size']
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.layernorm = nn.LayerNorm(config['hidden_size'])
        self.dropout = nn.Dropout(0.2)
    def forward(self, last_node_output, hidden_state):
        last_node_output = last_node_output
        last_node_output = self.dense(last_node_output)
        last_node_output = self.dropout(last_node_output)
        hidden_state = self.layernorm(last_node_output + hidden_state.transpose(1,2))
        hidden_state = hidden_state.transpose(1,2)
        return hidden_state   


class NonEdge(nn.Module):
    '''
    表示兩點之間沒連結
    '''
    def __init__(self):
        super(NonEdge, self).__init__()
    def forward(self, x):
        zero_x = x * 0
        return zero_x

class Conv1d_3gram(nn.Module):
    def __init__(self, config):
        super(Conv1d_3gram, self).__init__()
        self.hidden_size = config['hidden_size']
        self.conv_3gram = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.rc = ResidualConnect(config)
    def forward(self, x):
        '''
        x : (batch_size, hidden_dim, seq_len)
        output : (batch_size, hidden_dim, seq_len)
        '''
        output = self.conv_3gram(x)
        #adv_output = self.conv_3gram(adv_x)
        #output = torch.cat([output, adv_output], dim=1)
        # place hidden_dim last
        #output = self.dropout(output)
        output = self.layernorm(output.transpose(1,2))
        output = F.relu(output)
        
        rc_output = self.rc(output, x)
        #adv_rc_output = self.rc(output, adv_x)
        return rc_output
    
class Conv1d_5gram(nn.Module):
    def __init__(self, config):
        super(Conv1d_5gram, self).__init__()
        self.hidden_size = config['hidden_size']
        self.conv_5gram = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, stride=1, padding=2)
        #self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.rc = ResidualConnect(config)
    def forward(self, x):
        '''
        x : (batch_size, hidden_dim, seq_len)
        '''
        output = self.conv_5gram(x)
        #adv_output = self.conv_5gram(adv_x)
        #output = torch.cat([output, adv_output], dim=1)
        #output = self.dropout(output)
        # place hidden_dim last
        output = self.layernorm(output.transpose(1,2))
        output = F.relu(output)
        rc_output = self.rc(output, x)
        #adv_rc_output = self.rc(output, adv_x)
        return rc_output
    
class Conv1d_7gram(nn.Module):
    def __init__(self, config):
        super(Conv1d_7gram, self).__init__()
        self.hidden_size = config['hidden_size']
        self.conv_7gram = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=7, stride=1, padding=3)
        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.rc = ResidualConnect(config)
    def forward(self, x):
        '''
        x : (batch_size, hidden_dim, seq_len)
        '''
        output = self.conv_7gram(x)
        #adv_output = self.conv_7gram(adv_x)
        #output = torch.cat([output, adv_output], dim=1)
        #output = self.dropout(output)
        # place hidden_dim last
        output = self.layernorm(output.transpose(1,2))
        output = F.relu(output)
        rc_output = self.rc(output, x)
        #adv_rc_output = self.rc(output, adv_x)
        return rc_output
        
class Dil2_Conv1d(nn.Module):
    def __init__(self, config):
        super(Dil2_Conv1d, self).__init__()
        self.hidden_size = config['hidden_size']
        self.dil1_conv= nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.rc = ResidualConnect(config)
    def forward(self, x):
        '''
        x : (batch_size, hidden_dim, seq_len)
        '''
        output = self.dil1_conv(x) 
        #adv_output = self.dil1_conv(adv_x)
        #output = torch.cat([output, adv_output], dim=1)
        #output = self.dropout(output)
        # place hidden_dim last
        output = self.layernorm(output.transpose(1,2))
        output = F.relu(output)
        rc_output = self.rc(output, x)
        #adv_rc_output = self.rc(output, adv_x)
        return rc_output
                
class Dil4_Conv1d(nn.Module):
    def __init__(self, config):
        super(Dil4_Conv1d, self).__init__()
        self.hidden_size = config['hidden_size']
        self.dil2_conv= nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.rc = ResidualConnect(config)
    def forward(self, x):
        '''
        x : (batch_size, hidden_dim, seq_len)
        '''
        output = self.dil2_conv(x) 
        #adv_output = self.dil4_conv(adv_x)
        #output = torch.cat([output, adv_output], dim=1)
        #output = self.dropout(output)
        # place hidden_dim last
        output = self.layernorm(output.transpose(1,2))
        output = F.relu(output)
        rc_output = self.rc(output, x)
        #adv_rc_output = self.rc(output, adv_x)
        return rc_output

class Dil6_Conv1d(nn.Module):
    def __init__(self, config):
        super(Dil6_Conv1d, self).__init__()
        self.hidden_size = config['hidden_size']
        self.dil5_conv= nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=6, dilation=6)
        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.rc = ResidualConnect(config)
    def forward(self, x):
        '''
        x : (batch_size, hidden_dim, seq_len)
        '''
        output = self.dil5_conv(x) 
        #adv_output = self.dil6_conv(adv_x)
        #output = torch.cat([output, adv_output], dim=1)
        #output = self.dropout(output)
        # place hidden_dim last
        output = self.layernorm(output.transpose(1,2))
        output = F.relu(output)
        rc_output = self.rc(output, x)
        #adv_rc_output = self.rc(output, adv_x)
        return rc_output
    
class Max3_pooling1d(nn.Module):
    def __init__(self):
        super(Max3_pooling1d, self).__init__()
        self.pooling = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.pooling(x)
        #adv_x = self.pooling(adv_x)
        return x
    
class Max5_pooling1d(nn.Module):
    def __init__(self):
        super(Max5_pooling1d, self).__init__()
        self.pooling = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)
    def forward(self, x):
        x = self.pooling(x)
        #adv_x = self.pooling(adv_x)
        return x
    
class Max7_pooling1d(nn.Module):
    def __init__(self):
        super(Max7_pooling1d, self).__init__()
        self.pooling = nn.MaxPool1d(kernel_size=7, stride=1, padding=3)
    def forward(self, x):
        x = self.pooling(x)
        #adv_x = self.pooling(adv_x)
        return x
    
    
class Avg3Pooling1d(nn.Module):
    def __init__(self):
        super(Avg3Pooling1d, self).__init__()
        self.pooling = nn.AvgPool1d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
    def forward(self, x):
        x = self.pooling(x)
        #adv_x = self.pooling(adv_x)
        return x
    
class Avg5Pooling1d(nn.Module):
    def __init__(self):
        super(Avg5Pooling1d, self).__init__()
        self.pooling = nn.AvgPool1d(kernel_size=5, stride=1, padding=2, count_include_pad=False)
    def forward(self, x):
        x = self.pooling(x)
        #adv_x = self.pooling(adv_x)
        return x
    
class Avg7Pooling1d(nn.Module):
    def __init__(self):
        super(Avg7Pooling1d, self).__init__()
        self.pooling = nn.AvgPool1d(kernel_size=7, stride=1, padding=3, count_include_pad=False)
    def forward(self, x):
        x = self.pooling(x)
        #adv_x = self.pooling(adv_x)
        return x
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(config['hidden_size'], config['hidden_size'])
    def forward(self,x):
        output, _ = self.lstm(x)
        return output



    
operations_index = {0:"None", 1:"Conv1d_3gram",2:"Conv1d_5gram", 3:"Conv1d_7gram",
                    4:"Dil2_Conv1d", 5:"Dil4_Conv1d", 6:"Dil6_Conv1d", 7:"Max3_pooling1d",8:"Avg3Pooling1d"}


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = {
          'vocab_size': len(tokenizer.vocab),
          'hidden_size' : 256,
          'input_node_numbers':1,
          'intermediate_node_numbers' :6,
          'operation_numbers':len(operations_index), 
          'seq_len' : 128,
          'batch_size':64,
          'mask_number':20
         }

operations = {
    'None': NonEdge(),
    'Conv1d_3gram': Conv1d_3gram(config),
    'Conv1d_5gram': Conv1d_5gram(config),
    'Conv1d_7gram': Conv1d_7gram(config),
    'Dil2_Conv1d': Dil2_Conv1d(config),
    'Dil4_Conv1d': Dil4_Conv1d(config),
    'Dil6_Conv1d': Dil6_Conv1d(config),
    'Max3_pooling1d': Max3_pooling1d(),
    'Avg3Pooling1d':Avg3Pooling1d()    
}


def Mutual_Information(layer_embeddings, alpha):
    loss = 0
    first_loss = 0
    second_loss = 0
    eplison = 1e-7
    for layer_embedding in layer_embeddings:
        layer_embedding = layer_embedding.unsqueeze(0)
        Lx = nn.Softmax(dim=2)(layer_embedding)
        # do sample mean 
        ELx = torch.mean(Lx, dim=1)
        K = ELx.shape[1]
        first_term = torch.sum(-Lx * torch.log(Lx + eplison), dim=2).mean(1)

        second_term = -torch.sum(((1 / K) * torch.log(ELx + eplison) + \
                                  ((K -1) / K) * torch.log(1-ELx + eplison)), dim=1)
        #second_term = torch.sum(ELx * torch.log(ELx + eplison), dim=1)
        loss += first_term + second_term
        first_loss += first_term
        second_loss += second_term
    
    mean_layer_loss = (loss * alpha) / len(layer_embeddings)
    first_loss = first_loss / len(layer_embeddings)
    second_loss = second_loss / len(layer_embeddings)
    return mean_layer_loss, first_loss, second_loss



# 計算 node的加權操作
class WeightOperation(nn.Module):
    def __init__(self, operations):
        super(WeightOperation, self).__init__()
        self.operations = operations 
        self.ops = nn.ModuleList()
        for operation_name, operation in self.operations.items():
            self.ops.append(operation)
    def forward(self, x, operation_weight):
        
        weighted_op = sum(w*op(x) for w, op in zip(operation_weight, self.ops))
        
        return weighted_op
    
class Transform(nn.Module):
    def __init__(self, config):
        super(Transform, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.layernorm = nn.LayerNorm(config['hidden_size'])
    def forward(self, x):
        x = F.gelu(self.dense(x))
        x = self.layernorm(x)
        return x

class DiscreteLayer(nn.Module):
    def __init__(self, config, decay, eps):
        super().__init__()
        self.output_dim = config['hidden_size']
        self.cluster_size = 32
        self.decay = decay
        self.eps = eps
        self.embed = torch.randn(self.output_dim, self.cluster_size).cuda()
        self.register_buffer('cluster_number', torch.zeros(self.cluster_size))
        self.register_buffer('embed_avg', self.embed.clone())
    
    def forward(self, x, training):
        x = x.transpose(1,2)
        flatten = x.reshape(-1, self.output_dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed 
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, emb_ind = dist.min(dim=1)
        embed_onehot = F.one_hot(emb_ind, self.cluster_size).type(flatten.dtype)
        emb_ind = emb_ind.view(*x.shape[:-1])
        quantize = self.embed_code(emb_ind)
        
        if training:
            self.cluster_number.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_number.sum()
            cluster_number = (
                (self.cluster_number + self.eps) / (n + self.cluster_size * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_number.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)            
    
        discrete_loss = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()
        quantize = quantize.transpose(1,2)
        
        return quantize,  self.embed , discrete_loss
    
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))     
    
class NASMI(nn.Module):
    def __init__(self, config, class_numbers, device, training):
        super(NASMI, self).__init__()
        self.training = training
        self.hidden_size = config['hidden_size']
        self.seq_len = config['seq_len']
        self.batch_size = config['batch_size']
        self.input_node_numbers = config['input_node_numbers']
        self.intermediate_node_numbers = config['intermediate_node_numbers']
        self.operation_number = config['operation_numbers']
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layernorm = nn.LayerNorm(config['hidden_size'])
        self.dropout = nn.Dropout(0.2)
        # initial alpha
        self.edge_number = sum(1 for i in range(self.intermediate_node_numbers) for j in range(self.input_node_numbers+i))
        self.alpha = nn.Parameter(torch.randn(self.edge_number, self.operation_number))
        self.alpha_linear = nn.Linear(self.operation_number, self.operation_number)
        # every edge need select a operation
        self.weighted_op_list = nn.ModuleList()
        for i in range(self.intermediate_node_numbers):
            for j in range(self.input_node_numbers+i):
                weighted_op = WeightOperation(operations)
                self.weighted_op_list.append(weighted_op)
        
        #self.self_attention = SelfAttention(config['hidden_size'], 4, 64, 0.2)
        
        #self.classify_mask_layer = nn.Linear(config['hidden_size'], config['vocab_size'])
        self.discrete_layer_list = nn.ModuleList()
        for i in range(self.intermediate_node_numbers):
            self.discrete_layer_list.append(DiscreteLayer(config, decay=0.01, eps=1e-8))
        
        self.classification_layer = nn.Linear(self.hidden_size, class_numbers)
        
    def forward(self, sample,noise=False):
        '''
        input_node1、input_node2 : (batch_size, hidden_dim, seq_len)
        '''
        initial_embedding = []
        word_embedding = self.embedding(sample)
        if noise:
            word_embedding = self.dropout(word_embedding)
        initial_embedding.append(word_embedding)
        word_embedding = self.layernorm(word_embedding)
        input_node1 = word_embedding.transpose(1,2)
        
        alpha = self.alpha_linear(self.alpha)
        operation_weight = F.softmax(alpha, dim=1)
        
        #calculate each intermediate node output according nodes which connect with them
        offset = 0
        node_output = [input_node1]
        total_discrete_z = []
        total_layer_discrete_loss = 0
        for i in range(self.intermediate_node_numbers):
            intermediate_node_output = torch.zeros_like(node_output[0])
            for j, h in enumerate(node_output):
                intermediate_node_output += self.weighted_op_list[offset+j](h, operation_weight[offset+j])
            intermediate_node_output, discrete_z, discrete_loss = self.discrete_layer_list[i](intermediate_node_output, self.training)
            total_layer_discrete_loss += discrete_loss
            total_discrete_z.append(discrete_z)
            offset += len(node_output)
            node_output.append(intermediate_node_output)
        total_layer_discrete_loss = total_layer_discrete_loss / len(total_discrete_z)
        '''
        attn_node_output = []
        for output in node_output:
            output = self.self_attention(output)
            attn_node_output.append(output)
        '''   
        
        last_node_output = F.relu(torch.sum(node_output[-1], dim=2))
        
        last_node_output = self.classification_layer(last_node_output)
        
        return last_node_output, total_discrete_z, total_layer_discrete_loss
    
    def kmax_pooling(x, dim, k):
        index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
        return x.gather(dim, index)
    
