# regex-engine

A regular-expression engine built the way the classics are: a pattern is parsed
into an AST, compiled with **Thompson's construction** into an NFA, and matched
with a **parallel-state simulation** (a Pike VM that carries capture slots
along). Because every live state advances in lock-step over the input, the
engine **never backtracks** — `(a+)+b` on 30 `a`s finishes in microseconds
while a backtracking engine would take longer than the heat death of your
patience.

Leftmost-first (Python `re`) preference order is preserved: alternatives are
tried in source order and greedy quantifiers prefer longer matches, so results
agree with `re` on the curated differential corpus in `tests/`.

Zero dependencies — standard library only.

## Runnable example

```python
>>> import regex_engine as rx
>>> m = rx.search(r'(?P<user>\w+)@(?P<dom>\w+)\.(\w+)', 'mail: bob@example.com now')
>>> m.group('user'), m.group('dom'), m.group(3)
('bob', 'example', 'com')
>>> m.span()
(6, 21)
>>> rx.findall(r'\d+', 'a1b22c333')
['1', '22', '333']
>>> rx.sub(r'\d+', '#', 'a1b22c333')
'a#b#c#'
>>> rx.fullmatch(r'a+', 'aaa')
<Match span=(0, 3) match='aaa'>
>>> rx.match(r'a+', 'baa') is None
True
>>> [x.span() for x in rx.finditer(r'\d+', 'a1b22c333')]
[(1, 2), (3, 5), (6, 9)]
>>> rx.search(r'(a+)+b', 'a' * 30) is None   # no catastrophic backtracking
True
>>> rx.compile('(abc')
Traceback (most recent call last):
    ...
regex_engine.ParseError: missing ), unterminated subpattern at position 0
```

## Supported syntax

| Feature | Notes |
| --- | --- |
| literals, `.` | `.` matches any char except `\n` (`DOTALL` to include it) |
| `* + ?` | greedy by default; `*? +? ??` lazy forms |
| `{n}` `{n,}` `{n,m}` | counted repeats, lazy variants supported |
| `[abc] [a-z] [^...]` | ranges, negation, escapes inside, `]`/`-` first |
| `\d \w \s \D \W \S` | Unicode-aware, matching `re` |
| `\n \t \\ \. \+` etc. | standard escapes; unknown escapes are errors |
| `^ $` | anchors; `MULTILINE` per-line, `$` before final newline |
| `a\|b` alternation | leftmost-first preference |
| `(...)` `(?:...)` | capturing and non-capturing groups |
| `(?P<name>...)` | named groups, `m.group("name")` |
| flags | `IGNORECASE`, `MULTILINE`, `DOTALL` (values equal to `re`) |

Deliberately **not** supported — each raises `ParseError` with the offending
position: backreferences (`\1`), lookaround (`(?=`, `(?<!`), and the
zero-width escapes `\b \B \A \Z`.

## APIs

`compile(pattern, flags)` returns a `Regex` with `.search/.match/.fullmatch/
.findall/.finditer/.sub`, plus module-level one-shots: `search`, `match`,
`fullmatch`, `findall`, `finditer`, `sub`. `Match` objects expose `group`,
`groups`, `span`, `start`, `end`, `groupdict`, `__getitem__`, and `pos`.

## Tests

From the project root:

```
python -m unittest discover -s tests -v
```

The suite runs a curated differential corpus (100+ hand-picked cases, never
random) asserting agreement with Python's `re` on spans, groups, `findall` and
`sub` results, plus engine-specific behaviour: greedy/lazy preference,
capture slots (nested and unmatched), anchor semantics under `MULTILINE`,
`finditer` empty-match advancement, `ParseError` positions, and a
catastrophic-backtracking case that must complete in linear time.

## Files

- `regex_engine.py` — parser, Thompson compiler, Pike VM, match objects, APIs
- `tests/test_regex_engine.py` — differential corpus + engine behaviour tests
