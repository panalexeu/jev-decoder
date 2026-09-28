import numpy as np 
from rich import print 
from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient, Choice 

def _sample(probs: dict, t: float=1.0):
    chars = list(probs.keys())
    p = np.array(list(probs.values()), dtype=float)
    # softmax(log(p) / T) = exp(log(p) / T) / sum(...) = p^(1/T) / sum(...)
    p = p ** (1 / t)
    p /= p.sum()  
    return np.random.choice(chars, p=p)

def _form_ascii_dict(): 
    return {chr(i): chr(i) for i in range(128)}

if __name__ == '__main__': 
    load_dotenv()  
    client = TypeSafeClient() 

    instr = 'Choose the next ASCII character to generate a coherent continuation of the provided Shakespeare text.'
    ascii_choices = _form_ascii_dict()
    prefix = 'First Citizen:\nBefore we proceed any further' 
    ticket = prefix 

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
                    criteria=ascii_choices
                ) 
            } 
        ) 
        probs = response.answers['char'].probabilities
        char = _sample(probs, t)
        ticket += char

    print(ticket)
