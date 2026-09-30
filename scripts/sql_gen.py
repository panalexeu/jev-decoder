import json 
import sqlite3
from typing import Any
from collections import Counter

from lark import Lark
from rich import print 
from dotenv import load_dotenv
from lark.lexer import PatternStr
from typesafe_sdk import TypeSafeClient, Choice

_data_path = './data/spider_data/'
def _get_val_ds() -> dict: 
    # dev subset (1034 rows) is actually the same as validation subset (1.03k rows) here: https://huggingface.co/datasets/xlangai/spider
    subset = 'dev'
    path = _data_path + subset + '.json'  
    return json.load(open(path, 'r'))

_schemas_path = './data/spider_data/tables.json' 
def _load_schemas(): 
    schemas = {}
    for db in json.load(open(_schemas_path, 'r')):
        tables = db['table_names_original']
        cols = {t: [] for t in tables}
        for t_idx, c in db['column_names_original']: 
            if t_idx >= 0: cols[tables[t_idx]].append(c)
        schemas[db['db_id']] = cols
    return schemas

_grammar_path = './sql.lark'
def _get_grammar() -> str: 
    with open(_grammar_path, 'r') as f: return f.read() 

def _get_tokens(parser, query: str, ref_query: str, schema: dict) -> set:
    qp = parser.parse_interactive(query)
    qp.exhaust_lexer()
    ref = parser.parse_interactive(ref_query).exhaust_lexer()
    values = {
        "NAME": set(schema) | {c for cs in schema.values() for c in cs} | {f"T{i}" for i in range(1, 6)},
        "AGG_FN": {"COUNT", "SUM", "AVG", "MIN", "MAX"},
        "STRING": {t.value for t in ref if t.type == "STRING"},
        "NUMBER": {t.value for t in ref if t.type == "NUMBER"} or {"1"},
    }
    out = set()
    for t in qp.accepts() - {"$END"}:
        if t in values:
            out |= values[t]
        elif isinstance(p := parser.get_terminal(t).pattern, PatternStr):
            out.add(p.value)
    return out

def _get_criteria(tokens: tuple) -> dict: 
    return {t: t for t in tokens}

def _exec_statement(statement: str, db_id: str) -> list[Any] | None: 
    path = _data_path + 'database/' + f'{db_id}/' + f'{db_id}.sqlite'
    con = sqlite3.connect(path)
    try: 
        return con.execute(statement).fetchall()
    except Exception: 
        return None

if __name__ == "__main__":
    load_dotenv()
    ds = _get_val_ds() 
    parser = Lark(_get_grammar(), parser="lalr")
    client = TypeSafeClient() 
    schemas = _load_schemas() 

    prefix = 'Choose the next SQL token to generate an SQL query that will answer the question:'
    
    # sampling params 
    verbose = False
    max_tokens = 64
    END_TOKEN = ';'

    for row in ds: 
        schema = schemas[row['db_id']]
        instr = f'{prefix}{row['question']}\nSchema:\n{str(schema)}'
        ref_query = row['query']
        query = ''
        
        i = 0 
        while True: 
            tokens = _get_tokens(parser, query, ref_query, schema)
            response = client.system_one(
                state=query, 
                questions={
                    'token': Choice(
                        instructions=instr, 
                        criteria=_get_criteria(tokens)
                    )
                }
            )
            token = response.answers['token'].choice
            query = query + ' ' + token

            if verbose: 
                print('*' * 32)
                print(tokens)
                print(response.answers['token'].probabilities)

            i += 1
            if token == END_TOKEN or i >= max_tokens: 
                break
            
        print('=' * 32)
        print(f'question: {row['question']} | query: {query}')
        print('exec result:')
        print(_exec_statement(query, row['db_id']))
        input('press enter: ')
