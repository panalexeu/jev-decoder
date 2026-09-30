import json 

from lark import Lark
from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient, Choice

_data_path = './data/spider_data/'
def _get_val_ds() -> dict: 
    # dev subset (1034 rows) is actually the same as validation subset (1.03k rows) here: https://huggingface.co/datasets/xlangai/spider
    subset = 'dev'
    path = _data_path + subset + '.json'  
    with open(path, 'r') as f: 
        return json.load(f)

_grammar_path = './sql.lark'
def _get_grammar() -> str: 
    with open(_grammar_path, 'r') as f: return f.read() 

def _get_choices(tokens: list[str]) -> dict[str, str]: 
    return {token: token for token in tokens} 

if __name__ == "__main__":
    load_dotenv()
    ds = _get_val_ds() 
    parser = Lark(_get_grammar(), parser="lalr")
    client = TypeSafeClient() 

    prefix = 'Choose the next SQL token to generate an SQL query that will answer the question:'
    teacher_forcing = False 

    for row in ds: 
        instr = prefix + row['question']
        ref_query_parser = parser.parse_interactive(row['query'])
        ref_query_tokens = ref_query_parser.exhaust_lexer()
        query = ''
        for token in ref_query_tokens:
            token_type, token_value = token.type, token.value
            query_parser = parser.parse_interactive(query)
            query_parser.exhaust_lexer() 
            accept_tokens = query_parser.accepts() 
            # fucking retarded replacement in a set
            accept_tokens.discard(token_type)
            accept_tokens.add(token_value)

            response = client.system_one(
                state=query, 
                questions={
                    'token': Choice(
                        instructions=instr, 
                        criteria=_get_choices(accept_tokens)
                    )
                }
            )
            choice = response.answers['token'].choice

            if teacher_forcing: 
                query = query + ' ' + token_value
            else: 
                query = query + ' ' + choice 

        print(f'question: {row['question']} | query: {query}')
        breakpoint()
