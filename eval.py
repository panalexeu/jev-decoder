import numpy as np 
from rich import print 
from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient, Choice 

with open('./data/shk.txt', 'r') as f: 
    _content = f.read() 

def _form_shk_vocab(): 
    global _content
    chars = sorted(list(set(_content)))
    return {ch: ch for ch in chars}

_idx = -1
_block_size = 256 
def _next_sample(): 
    global _content
    global _idx 
    _idx += 1
    return _content[_idx:_block_size+_idx]

if __name__ == '__main__': 
    load_dotenv()  
    client = TypeSafeClient() 

    instr = 'Choose the next ASCII character to generate a coherent continuation of the provided Shakespeare text.'
    ascii_choices = _form_shk_vocab()
    ticket = _next_sample()

    # sampling params 
    tokens = 256
    t = 1.0
    
    # generation 
    losses = np.zeros(tokens)
    for i in range(tokens): 
        response = client.system_one(
            state=ticket,
            questions={
                'char': Choice(
                    instructions=instr, 
                    criteria=ascii_choices
                ) 
            } 
        ) 
        target = _content[_idx + _block_size]
        probs = response.answers['char'].probabilities 
        loss = -np.log(max(probs[target], 1e-12))
        losses[i] = loss
        ticket = _next_sample()
        print(f'target: {repr(target)}, loss: {loss:.2f}')

    print(f'avg. loss: {losses.mean():.2f}, uniform sampling: {np.log(len(ascii_choices.keys())):.2f}')
