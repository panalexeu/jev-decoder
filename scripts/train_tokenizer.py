import os 
import sentencepiece as spm

_data_path = './data/shk.txt'

if __name__ == '__main__': 
    if not os.path.exists(_data_path): raise IOError(f'{_data_path} is not found. Run `dwnld_shk.py` first.')
    
    with open(_data_path, 'r') as f: content = f.read()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(content.lower().splitlines()), # lower so more meaningful tokens are formed
        model_prefix="shk",
        vocab_size=255,  # the maximum amount of choices jev allows for `Choice` primitive is 255 hence 255 is the vocab size
        model_type="bpe",
    )
    sp = spm.SentencePieceProcessor(model_file='shk.model')
    tokens = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    print(tokens)
    