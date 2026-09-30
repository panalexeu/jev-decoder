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
    con = sqlite3.connect(f'{_data_path}database/{db_id}/{db_id}.sqlite')
    con.text_factory = lambda b: b.decode(errors='ignore')
    steps = [0]
    def abort():
        steps[0] += 1
        return steps[0] > 10_000  
    con.set_progress_handler(abort, 1000)  # called every 1000 VM instructions -> ~10M limit
    try:
        return con.execute(statement).fetchall()
    except Exception:
        return None
    finally:
        con.close()

def _eval_queries(query: str, ref_query: str, db_id: str) -> bool:
    res, ref = _exec_statement(query, db_id), _exec_statement(ref_query, db_id)
    return res is not None and ref is not None and Counter(res) == Counter(ref)

if __name__ == "__main__":
    load_dotenv()
    ds = _get_val_ds() 
    parser = Lark(_get_grammar(), parser="lalr")
    client = TypeSafeClient() 
    schemas = _load_schemas() 

    # sampling params 
    verbose = False
    max_tokens = 128
    END_TOKEN = ';'

    match = 0 
    for i, row in enumerate(ds): 
        schema = schemas[row['db_id']]
        instr = f'Choose the next SQL token to generate an SQL query that will answer the question:\n{row['question']}\nSchema:\n{str(schema)}'
        ref_query = row['query']
        query = ''

        # "autoregressive" geneeration here: 
        j = 0 
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

            j += 1
            if token == END_TOKEN or j >= max_tokens: 
                break

        if verbose: 
            print('=' * 32)
            print(f'question: {row['question']} | query: {query}')
            print('exec result:')
            print(_exec_statement(query, row['db_id']))

        
        res = _eval_queries(query, row['query'], row['db_id'])
        match += int(res)

        print(f'q: {i+1}/{len(ds)}, matches: {match}')
