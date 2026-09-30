import re

from lark import Lark
from lark.lexer import PatternStr
from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient, Choice

from sql_gen import _load_schemas, _get_criteria, _get_grammar, _exec_statement


def _get_tokens(parser, query: str, words: set[str], numbers: set[str], schema: dict) -> set:
    qp = parser.parse_interactive(query)
    qp.exhaust_lexer()
    values = {
        "NAME": set(schema) | {c for cs in schema.values() for c in cs} | {f"T{i}" for i in range(1, 6)},
        "AGG_FN": {"COUNT", "SUM", "AVG", "MIN", "MAX"},
        "STRING": words,
        "NUMBER": numbers or {"1"},
    }
    out = set()
    for t in qp.accepts() - {"$END"}:
        if t in values:
            out |= values[t]
        elif isinstance(p := parser.get_terminal(t).pattern, PatternStr):
            out.add(p.value)
    return out


if __name__ == '__main__':
    load_dotenv()
    parser = Lark(_get_grammar(), parser="lalr")
    client = TypeSafeClient()
    db_id = 'network_1'
    schema = _load_schemas()[db_id]

    print('db_schema: ', str(schema))
    prompt = input('Enter your prompt: ')
    words = {f'"{w}"' for w in re.findall(r'\b[A-Z]\w*', prompt)}
    numbers = set(re.findall(r'\b\d+\b', prompt))
    print('strings:', words, '| numbers:', numbers)

    instr = f'Choose the next SQL token to generate an SQL query that will answer the question:\n{prompt}\nSchema:\n{str(schema)}'

    # sampling params
    max_tokens = 128
    END_TOKEN = ';'

    query = ''
    i = 0
    while True:
        tokens = _get_tokens(parser, query, words, numbers, schema)
        if not tokens:
            break
        response = client.system_one(
            state=query,
            questions={'token': Choice(instructions=instr, criteria=_get_criteria(tokens))},
        )
        token = response.answers['token'].choice
        query = query + ' ' + token
        print('\r' + query, end='', flush=True)

        i += 1
        if token == END_TOKEN or i >= max_tokens:
            break

    print('\nexec result:')
    print(_exec_statement(query, db_id))
    