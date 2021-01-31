import re
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import BertTokenizer
from tqdm import tqdm

def clean_not_english_word(x):
    if x is not None:
        x = re.sub("[^a-zA-Z']+",' ',x).strip()
        return x 
    else:
        return None
    
def preprocess(data, max_length, device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    preprocess_sample = []
    for row in tqdm(data, desc='Preprocessing...'):
        text = row
        text = text.replace('<br />', '').strip()
        text = clean_not_english_word(text)
        tokenize_sentence = tokenizer.encode(text)
        if len(tokenize_sentence) > max_length:
            tokenize_sentence = tokenize_sentence[:max_length]
        preprocess_sample.append(torch.tensor(tokenize_sentence, device=device))
    preprocess_sample = pad_sequence(preprocess_sample, batch_first=True)
    return preprocess_sample