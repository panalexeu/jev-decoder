import numpy as np 
from rich import print 
from dotenv import load_dotenv
import sentencepiece as spm 
from typesafe_sdk import TypeSafeClient, Choice 

def _sample(probs: dict, t: float=1.0):
    chars = list(probs.keys())
    p = np.array(list(probs.values()), dtype=float)
    # softmax(log(p) / T) = exp(log(p) / T) / sum(...) = p^(1/T) / sum(...)
    p = p ** (1 / t)
    p /= p.sum()  
    return np.random.choice(chars, p=p)

def _form_token_dict(sp):
    return {sp.id_to_piece(i): sp.id_to_piece(i) for i in range(sp.get_piece_size())}

if __name__ == '__main__': 
    load_dotenv()  
    client = TypeSafeClient() 
    sp = spm.SentencePieceProcessor(model_file='./data/shk.model')

    instr = 'Choose the next token to generate a coherent continuation of the provided Shakespeare text. Underscore before token means space.'
    token_choices = _form_token_dict(sp) 
    prefix = 'first citizen:\nbefore we proceed any further'
    prefix_enc = sp.encode(prefix)
    prefix_enc_str = ''.join([sp.id_to_piece(i) for i in prefix_enc])
    ticket = prefix_enc_str

    # sampling params 
    tokens = 8
    t = 1.0

    # generation 
    for i in range(tokens): 
        response = client.system_one(
            state=ticket,
            questions={
                'char': Choice(
                    instructions=instr, 
                    criteria=token_choices
                ) 
            } 
        ) 
        probs = response.answers['char'].probabilities
        char = _sample(probs, t)
        ticket += char

    print(ticket)
