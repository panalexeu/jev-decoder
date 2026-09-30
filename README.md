## jev-demo

I was curious about the jev release and decided to play with it.
jev is known not to be a decoder model but rather some form of encoder. It is advertised as a model that evaluates a state and
returns typed answers and probabilities.
jev supports a `Choice` primitive, which returns probabilities over a specified set of choices.
After finding out about this primitive, I immediately had a bunch of dumb ideas, and I'm experimenting with them in this repo.

### usage

Before running the scripts, download tiny-shakespeare with `uv run ./scripts/dwnld_shk.py` and train the tokenizer with `uv run ./scripts/train_tokenizer.py`.
Aalo download spider from [here](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view) if you want to run `sql_gen.py` script; extract the archive into `./data dir`.

### experiments

Below are some completed experiments:

#### ascii_char_gen.py

The idea is to use jev as an ASCII character-level decoder model via the `Choice` primitive. ASCII symbols are provided as choices, and the state is concatenated with characters sampled from the returned character probability distribution.

Instruction: `Choose the next ASCII character to generate a coherent continuation of the provided Shakespeare text.`

State: `First Citizen:\nBefore we proceed any further`

Criteria:
```python
def _form_ascii_dict():
    return {chr(i): chr(i) for i in range(128)}
```

Results of generating 16 new symbols:
```text
First Citizen:
Before we proceed any further t h h h   h Ai u
```

#### token_gen.py

The idea is once again to use jev as a decoder model, but this time asking it to generate tokens. The `Choice` primitive accepts a maximum of 255 choices, so a BPE tokenizer with a vocab of 255 is trained on lowercased tiny-shakespeare and provided to `Choice` as criteria. The state is concatenated with tokens sampled from the returned probability distribution.

Instruction: `Choose the next token to generate a coherent continuation of the provided Shakespeare text. Underscore before token means space.`

State: `first citizen:\nbefore we proceed any further`

Criteria:
```python
def _form_token_dict(sp):
    return {sp.id_to_piece(i): sp.id_to_piece(i) for i in range(sp.get_piece_size())}
```

Results of generating 32 tokens:
```text
▁first▁citizen:▁before▁we▁proceed▁any▁further▁but▁king▁let▁letow,▁my▁from▁lord▁to▁he▁me▁king▁he▁let▁myhememe▁king▁and▁'▁'and'▁''▁but?▁king,▁what
```

#### char_eval.py

Evaluate the average loss by asking jev to predict the next character after a 256-character window, and compare it to the loss achieved by uniform random sampling over characters (spoiler: it's much worse than uniform sampling).
Results achieved on 256 consecutive predictions from the start of tiny-shakespeare:

Instruction: `Choose the next ASCII character to generate a coherent continuation of the provided Shakespeare text.`

State:
```python
_idx = -1
_block_size = 256
def _next_sample():
    global _content
    global _idx
    _idx += 1
    return _content[_idx:_block_size+_idx]
```

Criteria:
```python
def _form_shk_vocab():
    global _content
    chars = sorted(list(set(_content)))
    return {ch: ch for ch in chars}
```

Results:
```text
avg. loss: 7.71, uniform sampling: 4.17, catastrophic misses (>20 nats): 45
```

#### sql_gen.py

todo, probably the funniest one if it works out

### future work

* can criteria/instruction tuning actually make jev a better decoder model?