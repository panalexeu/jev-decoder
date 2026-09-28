## jev-demo

I was curious about the jev release and decided to play with it.
jev is known not to be a decoder model but rather some form of encoder. It is advertised as a model that evaluates a state and
returns typed answers and probabilities.
jev supports a `Choice` primitive, which returns probabilities over a specified set of choices.
After finding out about this primitive, I immediately had a bunch of dumb ideas, and I'm experimenting with them in this repo.
Below are some completed experiments:

### usage 

Before running scripts download tiny-shaekspere `uv run ./scripts/dwnld_shk.py` and train tokenizer `uv run ./scripts/train_tokenizer.py`. 

### experiments 

#### ascii_char_gen

#### token_gen 

#### eval 

