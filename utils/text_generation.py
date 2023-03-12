import constants as CONSTANTS
import torch
from random import choices


def generate_text(model,vocab,tokenizer,starting_text="",num_of_tokens_to_generate=50):
    
    text_str = starting_text                    # this is what we will display.
    text_tokens = vocab(tokenizer(text_str))    # this is what we give to the model.

    if len(text_tokens) < CONSTANTS.context_size:
        num_of_sos = CONSTANTS.context_size - len(text_tokens)
        text_tokens = vocab(["<SOS>"])*num_of_sos + text_tokens

    text_tokens = torch.tensor(text_tokens)

    for _ in range (num_of_tokens_to_generate):
        with torch.no_grad():
            dist = model(text_tokens[-CONSTANTS.context_size:].reshape(1,-1)).flatten().softmax(dim=0)

        clipped_dist = torch.clip(dist,min=0,max=0.1)
        dist = clipped_dist/sum(clipped_dist)
        # select a word from dist
        generated_token = choices(range(len(dist)),dist,k=1)[0]

        text_tokens = torch.cat((text_tokens,torch.tensor([generated_token])))
        text_str += f" {vocab.lookup_token(generated_token)}"
        
        
        
    return text_str