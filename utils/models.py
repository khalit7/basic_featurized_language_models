import constants as CONSTANTS

import torch.nn as nn
import torch




class LogLinearLM(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size,vocab_size)
        
        self.bias = nn.Embedding(1,vocab_size)
        
        self.bias_idx = torch.tensor([0])
        
    def forward(self,x):
        '''
        x has the shape (batch_size*context_size)
        output has the shape batch_size*vocab_size
        '''
        
        x = self.embed(x)                                           # shape is (batch_size*context_size*vocab_size)
        words_sum = x.sum(axis=1)                                   # shape is (batch_size*vocab_size)
        
        output = words_sum + self.bias( self.bias_idx ).reshape(-1) # shape is (batch_size*vocab_size)
        
        return output
    
    
    
class FFLM(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size,100)
        
        self.lin = nn.Linear(100*CONSTANTS.context_size,vocab_size)
        
        self.relu = nn.ReLU()
        
    def forward(self,x):
        '''
        x has the shape (batch_size*context_size)
        output has the shape batch_size*vocab_size
        '''
        
        x = self.embed(x)                                           # shape is (batch_size*context_size*vocab_size)
        b,c,v = x.shape
        x = x.reshape(b,c*v)                                        # shape is (batch_size*context_sizeXvocab_size)
        
        x = self.lin(x)                                             # shape is (batch_size*vocab_size)
        x = self.relu(x)                                           # shape is (batch_size*vocab_size)
        
        return x
    
    
    
