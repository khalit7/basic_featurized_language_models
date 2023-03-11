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
    
    
    
    
