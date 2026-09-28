import os 

import requests
import sentencepiece as spm

_data_dir = './data/'
_data_path = _data_dir + 'shk.txt'
_data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt'

if __name__ == '__main__': 
    if not os.path.exists(_data_dir): os.mkdir(_data_dir)
    if not os.path.exists(_data_path):
        bytes_ = requests.get(_data_url).content
        content = bytes_.decode('utf-8').lower()  # we have only 255 tokens 
        with open(_data_path, 'w') as f: f.write(content)

    spm.SentencePieceTrainer.train(
        input=_data_path,
        model_prefix="shk",
        vocab_size=255,
        model_type="bpe",
    )
    sp = spm.SentencePieceProcessor(model_file='shk.model')
    tokens = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    print(tokens)
    