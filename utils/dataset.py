import constants as CONSTANTS

import torch
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2
from torchtext.vocab import vocab
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer

from functools import partial

def _get_data_itr(split):
    data_itr = WikiText2(split=split)
    
    return data_itr


def _get_tokenizer():
    
    tokenizer = get_tokenizer("basic_english")
    
    return tokenizer

def _build_vocab(data_itr,tokenizer):
    v = build_vocab_from_iterator(map(tokenizer,data_itr),min_freq=CONSTANTS.min_freq,specials=["<unk>","<SOS>","<EOS>"])
    v.set_default_index(v["<unk>"])
    
    return v


def _collate_fn(batch,vocab,tokenizer):
    '''
    takes a batch of paragraphs from the dataset, returns a batch of X,Y
    '''
    X=[]
    Y=[]
    for b in batch:
        
        transformed_b = vocab(tokenizer(b))
        # if b is too short, skip it
        if len(transformed_b) < CONSTANTS.min_seq_len :
            continue
        # if b is too long, truncate it
        if len(transformed_b) > CONSTANTS.max_seq_len:
            b = b[0:CONSTANTS.max_seq_len]
        
        # add start of sentance and end of sentance
        sos = vocab(["<SOS>"]*CONSTANTS.context_size)
        eos = vocab(["<EOS>"]*CONSTANTS.context_size)
        transformed_b = sos + transformed_b + eos
        
        for i in range(CONSTANTS.context_size,len(transformed_b)):
            token_to_pred = transformed_b[i]
            context_tokens = transformed_b[i-CONSTANTS.context_size:i]
            X.append(context_tokens)
            Y.append(token_to_pred)
    X = torch.tensor(X,dtype=torch.long)
    Y = torch.tensor(Y,dtype=torch.long)
    
    return X,Y


def get_data_loader_and_vocab(data_split,batch_size=1,shuffle=True,vocab=None):
    
    # get data iterator:
    data_itr = _get_data_itr(data_split)
    
    # get tokenizer
    tokenizer = _get_tokenizer()
    
    # build vocab
    if vocab is None:
        vocab = _build_vocab(data_itr,tokenizer)
    
    dataloader = DataLoader(data_itr, batch_size=batch_size, shuffle=shuffle,collate_fn= partial(_collate_fn,vocab=vocab,tokenizer=tokenizer))
    
    return dataloader,vocab