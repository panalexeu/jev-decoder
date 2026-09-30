import re

from lark import Lark
from lark.lexer import PatternStr
from dotenv import load_dotenv
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from typesafe_sdk import TypeSafeClient, Choice

from sql_gen import _load_schemas, _get_criteria, _get_grammar, _exec_statement

console = Console()
TOP_K = 8


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


def _schema_view(schema):
    t = Table(box=box.SIMPLE, header_style='bold cyan')
    t.add_column('table')
    t.add_column('columns')
    for name, cols in schema.items():
        t.add_row(name, ', '.join(cols))
    return Panel(t, title='schema', border_style='blue')


def _values_view(words, numbers):
    return Text(f'strings: {", ".join(sorted(words)) or "-"}   numbers: {", ".join(sorted(numbers)) or "-"}', style='dim')


def _step_view(query, chosen, probs, step, n_opts):
    q = Text(query.strip(), style='white')
    q.append(' ' + chosen, style='bold black on green')
    q.append(' ▌', style='bold cyan')
    table = Table(box=None, expand=True, header_style='dim')
    table.add_column('option', no_wrap=True)
    table.add_column('probability', ratio=1)
    table.add_column('%', justify='right', width=7)
    for opt, p in sorted(probs.items(), key=lambda x: -x[1])[:TOP_K]:
        hit = opt == chosen
        table.add_row(
            Text(opt, style='bold green' if hit else ''),
            Text('█' * max(1, round(p * 40)), style='green' if hit else 'blue'),
            f'{p:.1%}',
        )
    return Group(
        Panel(q, title=f'[cyan]query[/] · step {step}', border_style='cyan'),
        Panel(table, title=f'top {min(TOP_K, n_opts)} of {n_opts} options', border_style='magenta'),
    )


def _result_view(query, rows):
    console.print(Panel(Syntax(query.strip(), 'sql', word_wrap=True, background_color='default'),
                        title='final query', border_style='green'))
    if rows is None:
        console.print(Panel('query failed to execute', title='result', border_style='red'))
        return
    t = Table(box=box.ROUNDED, header_style='bold', title=f'{len(rows)} row(s)')
    for k in range(len(rows[0]) if rows else 1):
        t.add_column(f'#{k + 1}')
    for r in rows[:20]:
        t.add_row(*map(str, r))
    if len(rows) > 20:
        t.caption = f'... {len(rows) - 20} more'
    console.print(t)


def _generate(parser, client, prompt, schema, db_id, max_tokens=128, end_token=';'):
    words = {f'"{w}"' for w in re.findall(r'\b[A-Z]\w*', prompt)}
    numbers = set(re.findall(r'\b\d+', prompt))
    console.print(_values_view(words, numbers))

    instr = f'Choose the next SQL token to generate an SQL query that will answer the question:\n{prompt}\nSchema:\n{str(schema)}'

    query = ''
    i = 0
    with Live(console=console, refresh_per_second=12) as live:
        while True:
            tokens = _get_tokens(parser, query, words, numbers, schema)
            if not tokens:
                break
            response = client.system_one(
                state=query,
                questions={'token': Choice(instructions=instr, criteria=_get_criteria(tokens))},
            )
            answer = response.answers['token']
            token = answer.choice
            probs = dict(answer.probabilities)

            i += 1
            live.update(_step_view(query, token, probs, i, len(tokens)))
            query = query + ' ' + token

            if token == end_token or i >= max_tokens:
                break

    console.print()
    _result_view(query, _exec_statement(query, db_id))


if __name__ == '__main__':
    load_dotenv()
    parser = Lark(_get_grammar(), parser="lalr")
    client = TypeSafeClient()
    db_id = 'network_1'
    schema = _load_schemas()[db_id]

    console.rule(f'[bold cyan]text-to-SQL · {db_id}')
    console.print(_schema_view(schema))
    console.print('[dim]empty line or "exit" to quit[/]\n')

    while True:
        try:
            prompt = console.input('[bold]Enter your prompt:[/] ').strip()
        except (KeyboardInterrupt, EOFError):
            break
        if prompt.lower() in ('', 'exit', 'quit'):
            break
        try:
            _generate(parser, client, prompt, schema, db_id)
        except KeyboardInterrupt:
            console.print('\n[yellow]generation stopped[/]')
        console.rule(style='dim')
        