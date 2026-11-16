"""regex-engine — regular expressions via Thompson NFA (never backtracks).

Parses a Python-compatible regex subset, compiles it to a Thompson NFA, and
simulates all alternatives in parallel with a Pike VM that preserves
leftmost-first (Perl-like) preference order — so results agree with Python's
``re`` without exponential backtracking.

Supported: literals, ``.``, greedy/lazy ``* + ? {n,m}``, character classes
(ranges, negation, set escapes), ``\\d \\D \\w \\W \\s \\S \\n \\t \\r \\\\``,
anchors ``^ $`` (multiline-aware), alternation ``|``, capturing and
non-capturing groups, ``IGNORECASE``/``MULTILINE``/``DOTALL``.

Not supported (documented): backreferences, lookarounds, ``\\b``/``\\B``,
conditional groups, atomic groups.  ``search``, ``match``, ``fullmatch``,
``sub``, ``findall`` and ``finditer`` are implemented; invalid patterns raise
``ParseError`` with a position.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "ParseError",
    "Regex",
    "Match",
    "compile",
    "search",
    "match",
    "fullmatch",
    "sub",
    "findall",
    "finditer",
    "IGNORECASE",
    "MULTILINE",
    "DOTALL",
    "I",
    "M",
    "S",
]

# Flag values match the ``re`` module so tests can share constants.
IGNORECASE = 2
MULTILINE = 8
DOTALL = 16
I = IGNORECASE
M = MULTILINE
S = DOTALL

MAX_CODEPOINT = 0x10FFFF


class ParseError(ValueError):
    """Invalid regular expression; carries ``.pos`` (0-based) and a message."""

    def __init__(self, message: str, pos: int) -> None:
        super().__init__(f"{message} at position {pos}")
        self.message = message
        self.pos = pos


# ------------------------------------------------------- character ranges --


def normalize(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort and merge overlapping/adjacent inclusive ranges."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for lo, hi in ranges[1:]:
        plo, phi = merged[-1]
        if lo <= phi + 1:
            merged[-1] = (plo, max(phi, hi))
        else:
            merged.append((lo, hi))
    return merged


def union(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return normalize(list(a) + list(b))


def complement(a: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Complement over ``0..MAX_CODEPOINT``."""
    out: List[Tuple[int, int]] = []
    cursor = 0
    for lo, hi in normalize(list(a)):
        if lo > cursor:
            out.append((cursor, lo - 1))
        cursor = max(cursor, hi + 1)
    if cursor <= MAX_CODEPOINT:
        out.append((cursor, MAX_CODEPOINT))
    return out


def in_ranges(ranges: Sequence[Tuple[int, int]], cp: int) -> bool:
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
        if cp < lo:
            return False
    return False


_DIGIT_RANGES = [(ord("0"), ord("9"))]
_ASCII_WORD = [
    (ord("0"), ord("9")),
    (ord("A"), ord("Z")),
    (ord("_"), ord("_")),
    (ord("a"), ord("z")),
]

_WORD_RANGES: Optional[List[Tuple[int, int]]] = None
_SPACE_RANGES: Optional[List[Tuple[int, int]]] = None
_DIGIT_RANGES_FULL: Optional[List[Tuple[int, int]]] = None


def _build_ranges(pred) -> List[Tuple[int, int]]:
    """Build ranges of codepoints where ``pred(chr(cp))`` is true."""
    ranges: List[Tuple[int, int]] = []
    start: Optional[int] = None
    prev = -1
    for cp in range(MAX_CODEPOINT + 1):
        if pred(chr(cp)):
            if start is None:
                start = cp
            prev = cp
        else:
            if start is not None:
                ranges.append((start, prev))
                start = None
    if start is not None:
        ranges.append((start, prev))
    return ranges


def digit_ranges() -> List[Tuple[int, int]]:
    """Unicode decimal digits (lazily built, mirrors ``str.isdigit`` closely)."""
    global _DIGIT_RANGES_FULL
    if _DIGIT_RANGES_FULL is None:
        _DIGIT_RANGES_FULL = normalize(_build_ranges(lambda c: c.isdigit()))
    return _DIGIT_RANGES_FULL


def word_ranges() -> List[Tuple[int, int]]:
    """Word characters: ``str.isalnum()`` plus underscore (mirrors ``\\w``)."""
    global _WORD_RANGES
    if _WORD_RANGES is None:
        _WORD_RANGES = normalize(_build_ranges(lambda c: c.isalnum() or c == "_"))
    return _WORD_RANGES


def space_ranges() -> List[Tuple[int, int]]:
    """Whitespace (mirrors ``str.isspace``, which ``\\s`` tracks closely)."""
    global _SPACE_RANGES
    if _SPACE_RANGES is None:
        _SPACE_RANGES = normalize(_build_ranges(lambda c: c.isspace()))
    return _SPACE_RANGES


# ------------------------------------------------------------------ AST ----


class _Alt:
    __slots__ = ("branches",)

    def __init__(self, branches):
        self.branches = branches


class _Concat:
    __slots__ = ("items",)

    def __init__(self, items):
        self.items = items


class _Repeat:
    __slots__ = ("node", "min", "max", "greedy")

    def __init__(self, node, min_, max_, greedy):
        self.node = node
        self.min = min_
        self.max = max_
        self.greedy = greedy


class _Char:
    __slots__ = ("ranges", "negated")

    def __init__(self, ranges, negated=False):
        self.ranges = ranges
        self.negated = negated


class _Group:
    __slots__ = ("node", "index")

    def __init__(self, node, index):
        self.node = node
        self.index = index  # 0-based slot pair index; None = non-capturing


class _Anchor:
    __slots__ = ("kind",)

    def __init__(self, kind):
        self.kind = kind  # 'start' or 'end'


# ---------------------------------------------------------------- parser ----


class _Parser:
    """Recursive-descent parser for the regex grammar."""

    def __init__(self, pattern: str, flags: int):
        self.pat = pattern
        self.pos = 0
        self.flags = flags
        self.ngroups = 0
        self.group_names: Dict[str, int] = {}

    def error(self, message: str, pos: Optional[int] = None) -> ParseError:
        return ParseError(message, self.pos if pos is None else pos)

    def parse(self):
        if not self.pat:
            # Empty pattern matches the empty string (same as Python).
            return _Concat([]), 0
        node = self.parse_alt()
        if self.pos < len(self.pat):
            ch = self.pat[self.pos]
            if ch == ")":
                raise self.error("unbalanced parenthesis", self.pos)
            raise self.error(f"unexpected character '{ch}'", self.pos)
        return node, self.ngroups

    def parse_alt(self):
        branches = [self.parse_concat()]
        while self.pos < len(self.pat) and self.pat[self.pos] == "|":
            self.pos += 1
            branches.append(self.parse_concat())
        if len(branches) == 1:
            return branches[0]
        return _Alt(branches)

    def parse_concat(self):
        items = []
        while self.pos < len(self.pat):
            ch = self.pat[self.pos]
            if ch in "|)":
                break
            if ch in "*+?":
                raise self.error("nothing to repeat", self.pos)
            items.append(self.parse_repeat())
        return _Concat(items)

    def parse_repeat(self):
        atom = self.parse_atom()
        while self.pos < len(self.pat) and self.pat[self.pos] in "*+?{":
            op_pos = self.pos
            ch = self.pat[self.pos]
            if ch == "{":
                quant = self.try_parse_brace()
                if quant is None:
                    break  # literal '{' when not a valid quantifier
            else:
                self.pos += 1
                quant = {"*": (0, None), "+": (1, None), "?": (0, 1)}[ch]
            min_, max_ = quant
            if max_ is not None and max_ < min_:
                raise self.error("min repeat greater than max repeat", op_pos)
            greedy = True
            if self.pos < len(self.pat) and self.pat[self.pos] == "?":
                greedy = False
                self.pos += 1
            if self.pos < len(self.pat) and self.pat[self.pos] in "*+{":
                raise self.error("multiple repeat", self.pos)
            atom = _Repeat(atom, min_, max_, greedy)
        return atom

    def try_parse_brace(self) -> Optional[Tuple[int, Optional[int]]]:
        """Parse ``{n}``, ``{n,}``, ``{n,m}``; return None if not quantifier-like."""
        start = self.pos
        i = self.pos + 1
        digits1 = i
        while i < len(self.pat) and self.pat[i].isdigit():
            i += 1
        if i == digits1:
            return None
        n = int(self.pat[digits1:i])
        if i < len(self.pat) and self.pat[i] == "}":
            self.pos = i + 1
            return (n, n)
        if i >= len(self.pat) or self.pat[i] != ",":
            return None
        i += 1
        if i < len(self.pat) and self.pat[i] == "}":
            self.pos = i + 1
            return (n, None)
        digits2 = i
        while i < len(self.pat) and self.pat[i].isdigit():
            i += 1
        if i == digits2 or i >= len(self.pat) or self.pat[i] != "}":
            return None
        m = int(self.pat[digits2:i])
        self.pos = i + 1
        return (n, m)

    def parse_atom(self):
        ch = self.pat[self.pos]
        if ch == "(":
            return self.parse_group()
        if ch == "[":
            return self.parse_class()
        if ch == ".":
            self.pos += 1
            if self.flags & DOTALL:
                return _Char([(0, MAX_CODEPOINT)])
            return _Char([(0, 9), (11, MAX_CODEPOINT)])  # everything but \n
        if ch == "^":
            self.pos += 1
            return _Anchor("start")
        if ch == "$":
            self.pos += 1
            return _Anchor("end")
        if ch == "\\":
            return self.parse_escape(in_class=False)
        if ch in "*+?":
            raise self.error("nothing to repeat", self.pos)
        if ch == ")":
            raise self.error("unbalanced parenthesis", self.pos)
        self.pos += 1
        return _Char([(ord(ch), ord(ch))])

    def parse_group(self):
        open_pos = self.pos
        self.pos += 1  # '('
        index: Optional[int] = None
        if self.pat.startswith("?:", self.pos):
            self.pos += 2
        elif self.pat.startswith("?P<", self.pos) or self.pat.startswith("?P=", self.pos):
            if self.pat.startswith("?P=", self.pos):
                raise self.error("named backreferences are not supported", open_pos)
            close = self.pat.find(">", self.pos + 3)
            if close < 0:
                raise self.error("unterminated group name", open_pos)
            name = self.pat[self.pos + 3 : close]
            if not name:
                raise self.error("missing group name", open_pos)
            if name in self.group_names:
                raise self.error(f"duplicate group name '{name}'", open_pos)
            self.group_names[name] = self.ngroups
            index = self.ngroups
            self.ngroups += 1
            self.pos = close + 1
        elif self.pat.startswith("?(", self.pos):
            raise self.error("conditional groups are not supported", open_pos)
        elif self.pat.startswith("?<=", self.pos) or self.pat.startswith("?<!", self.pos):
            raise self.error("lookbehind is not supported", open_pos)
        elif self.pat.startswith("?=", self.pos) or self.pat.startswith("?!", self.pos):
            raise self.error("lookahead is not supported", open_pos)
        elif self.pos < len(self.pat) and self.pat[self.pos] == "?":
            raise self.error(f"unsupported group extension '?{self.pat[self.pos + 1] if self.pos + 1 < len(self.pat) else ''}'", open_pos)
        else:
            index = self.ngroups
            self.ngroups += 1
        node = self.parse_alt()
        if self.pos >= len(self.pat) or self.pat[self.pos] != ")":
            raise self.error("missing ), unterminated subpattern", open_pos)
        self.pos += 1
        return _Group(node, index)

    def parse_class(self):
        open_pos = self.pos
        self.pos += 1  # '['
        negated = False
        if self.pos < len(self.pat) and self.pat[self.pos] == "^":
            negated = True
            self.pos += 1
        ranges: List[Tuple[int, int]] = []
        first = True
        while True:
            if self.pos >= len(self.pat):
                raise self.error("unterminated character set", open_pos)
            ch = self.pat[self.pos]
            if ch == "]" and not first:
                self.pos += 1
                break
            first = False
            if ch == "\\":
                esc = self.parse_escape(in_class=True)
                if esc.negated:
                    ranges = union(ranges, complement(esc.ranges))
                else:
                    ranges = union(ranges, esc.ranges)
                continue
            self.pos += 1
            lo = ord(ch)
            hi = lo
            if (
                self.pos + 1 < len(self.pat)
                and self.pat[self.pos] == "-"
                and self.pat[self.pos + 1] != "]"
            ):
                self.pos += 1
                hi_ch = self.pat[self.pos]
                if hi_ch == "\\":
                    esc = self.parse_escape(in_class=True)
                    if not isinstance(esc, _Char) or esc.negated:
                        raise self.error("invalid range endpoint", self.pos)
                    hi = esc.ranges[0][0]
                    if len(esc.ranges) != 1 or esc.ranges[0][0] != esc.ranges[0][1]:
                        hi = esc.ranges[-1][1]
                else:
                    self.pos += 1
                    hi = ord(hi_ch)
                if hi < lo:
                    raise self.error("bad character range", self.pos)
            ranges.append((lo, hi))
        return _Char(normalize(ranges), negated)

    def parse_escape(self, in_class: bool):
        esc_pos = self.pos
        self.pos += 1  # backslash
        if self.pos >= len(self.pat):
            raise self.error("incomplete escape", esc_pos)
        ch = self.pat[self.pos]
        self.pos += 1
        simple = {
            "n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "a": "\a", "0": "\0",
        }
        if ch in simple:
            cp = ord(simple[ch])
            return _Char([(cp, cp)])
        if ch == "x":
            hexs = self.pat[self.pos : self.pos + 2]
            if len(hexs) != 2 or any(c not in "0123456789abcdefABCDEF" for c in hexs):
                raise self.error("bad \\x escape", esc_pos)
            self.pos += 2
            cp = int(hexs, 16)
            return _Char([(cp, cp)])
        if ch == "u":
            hexs = self.pat[self.pos : self.pos + 4]
            if len(hexs) != 4 or any(c not in "0123456789abcdefABCDEF" for c in hexs):
                raise self.error("bad \\u escape", esc_pos)
            self.pos += 4
            cp = int(hexs, 16)
            return _Char([(cp, cp)])
        if ch == "d":
            return _Char(digit_ranges())
        if ch == "D":
            return _Char(complement(digit_ranges()))
        if ch == "w":
            return _Char(word_ranges())
        if ch == "W":
            return _Char(complement(word_ranges()))
        if ch == "s":
            return _Char(space_ranges())
        if ch == "S":
            return _Char(complement(space_ranges()))
        if ch in "bB":
            raise self.error(f"\\{ch} (word boundary) is not supported", esc_pos)
        if ch in "AZzG":
            raise self.error(f"\\{ch} is not supported", esc_pos)
        if ch.isalnum():
            raise self.error(f"bad escape '\\{ch}'", esc_pos)
        # Punctuation escapes are literals.
        cp = ord(ch)
        return _Char([(cp, cp)])


# --------------------------------------------------------------- compiler ---


class _Inst:
    """One NFA instruction: ``op`` plus up to two operands."""

    __slots__ = ("op", "a", "b")

    def __init__(self, op: str, a: Any = None, b: Any = None) -> None:
        self.op = op
        self.a = a
        self.b = b


class _Program:
    __slots__ = ("code", "ngroups", "icase", "multiline", "pattern")

    def __init__(self, code, ngroups, icase, multiline, pattern):
        self.code = code
        self.ngroups = ngroups
        self.icase = icase
        self.multiline = multiline
        self.pattern = pattern


def _build(ast, ngroups: int, icase: bool, multiline: bool, pattern: str) -> _Program:
    code: List[_Inst] = []

    def emit(op, a=None, b=None) -> int:
        code.append(_Inst(op, a, b))
        return len(code) - 1

    def build(node) -> None:
        if isinstance(node, _Concat):
            for item in node.items:
                build(item)
        elif isinstance(node, _Alt):
            jmps: List[int] = []
            for branch in node.branches[:-1]:
                split = emit("split")
                branch_start = len(code)
                build(branch)
                jmps.append(emit("jmp"))
                code[split] = _Inst("split", branch_start, len(code))
            build(node.branches[-1])
            end = len(code)
            for j in jmps:
                code[j] = _Inst("jmp", end)
        elif isinstance(node, _Repeat):
            build_repeat(node)
        elif isinstance(node, _Char):
            emit("char", node.ranges, node.negated)
        elif isinstance(node, _Anchor):
            emit("anchor", node.kind)
        elif isinstance(node, _Group):
            if node.index is not None:
                # Slots 0/1 belong to the overall match; groups start at 2.
                emit("save", 2 * node.index + 2)
                build(node.node)
                emit("save", 2 * node.index + 3)
            else:
                build(node.node)
        else:  # pragma: no cover - exhaustive over AST nodes
            raise ParseError(f"cannot compile {type(node).__name__}", 0)

    def build_repeat(r: _Repeat) -> None:
        if r.max == 0:
            return  # {0} matches the empty string only

        def star(body, greedy: bool) -> None:
            split = emit("split")
            body_start = len(code)
            build(body)
            emit("jmp", split)
            exit_at = len(code)
            if greedy:
                code[split] = _Inst("split", body_start, exit_at)
            else:
                code[split] = _Inst("split", exit_at, body_start)

        def optional(body, greedy: bool) -> None:
            split = emit("split")
            body_start = len(code)
            build(body)
            exit_at = len(code)
            if greedy:
                code[split] = _Inst("split", body_start, exit_at)
            else:
                code[split] = _Inst("split", exit_at, body_start)

        if r.max is None:
            for _ in range(r.min):
                build(r.node)
            star(r.node, r.greedy)
        else:
            for _ in range(r.min):
                build(r.node)
            for _ in range(r.max - r.min):
                optional(r.node, r.greedy)

    emit("save", 0)
    build(ast)
    emit("save", 1)
    emit("match")
    return _Program(code, ngroups, icase, multiline, pattern)


def _match_char(ranges, negated: bool, cp: int, icase: bool) -> bool:
    hit = in_ranges(ranges, cp)
    if not hit and icase:
        ch = chr(cp)
        for variant in (ch.lower(), ch.upper()):
            if variant != ch and in_ranges(ranges, ord(variant)):
                hit = True
                break
    return hit != negated


# ------------------------------------------------------------- Pike VM -----


class _Thread:
    __slots__ = ("pc", "caps")

    def __init__(self, pc: int, caps: List[int]) -> None:
        self.pc = pc
        self.caps = caps


def _add_thread(
    code: List[_Inst],
    pc: int,
    caps: List[int],
    out: List[_Thread],
    visited: set,
    anchor_pos: int,
    text: str,
    multiline: bool,
) -> None:
    """Epsilon-closure at ``anchor_pos``; threads leave ordered by preference.

    Depth-first, first-alternative-first — the same order a backtracking
    engine would try — so the simulation is leftmost-first compatible.
    """
    stack: List[Tuple[int, List[int]]] = [(pc, caps)]
    while stack:
        pc, caps = stack.pop()
        if pc in visited:
            continue
        visited.add(pc)
        inst = code[pc]
        op = inst.op
        if op == "save":
            new_caps = caps[:]
            new_caps[inst.a] = anchor_pos
            stack.append((pc + 1, new_caps))
        elif op == "split":
            stack.append((inst.b, caps))
            stack.append((inst.a, caps))
        elif op == "jmp":
            stack.append((inst.a, caps))
        elif op == "anchor":
            if inst.a == "start":
                ok = anchor_pos == 0 or (multiline and anchor_pos > 0 and text[anchor_pos - 1] == "\n")
            else:
                if multiline:
                    ok = anchor_pos == len(text) or text[anchor_pos] == "\n"
                else:
                    ok = anchor_pos == len(text) or (
                        anchor_pos == len(text) - 1 and text[anchor_pos] == "\n"
                    )
            if ok:
                stack.append((pc + 1, caps))
        else:  # 'char' or 'match': the thread is ready at this position
            out.append(_Thread(pc, caps))


def _simulate(prog: _Program, text: str, startpos: int, mode: str) -> Optional[List[int]]:
    """Run the parallel-state simulation.

    ``mode`` is ``'search'`` (leftmost anywhere), ``'match'`` (anchored at
    ``startpos``) or ``'full'`` (anchored, must consume the whole string).
    """
    code = prog.code
    n = len(text)
    startpos = max(0, min(startpos, n))
    caps_size = 2 * (prog.ngroups + 2)  # slots 0/1 = whole match, 2i+2/2i+3 = group i
    multiline = prog.multiline
    icase = prog.icase
    clist: List[_Thread] = []
    matched: Optional[List[int]] = None
    pos = startpos
    while pos <= n:
        if matched is None and (mode == "search" or pos == startpos):
            visited = {t.pc for t in clist}
            _add_thread(code, 0, [-1] * caps_size, clist, visited, pos, text, multiline)
        nlist: List[_Thread] = []
        spawn_visited: set = set()
        i = 0
        stop = False
        while i < len(clist):
            thread = clist[i]
            i += 1
            inst = code[thread.pc]
            if inst.op == "char":
                if pos < n and _match_char(inst.a, inst.b, ord(text[pos]), icase):
                    _add_thread(
                        code,
                        thread.pc + 1,
                        thread.caps,
                        nlist,
                        spawn_visited,
                        pos + 1,
                        text,
                        multiline,
                    )
            else:  # 'match'
                if mode == "full" and pos != n:
                    continue  # not a whole-string match: this thread dies
                # Everything still alive in clist after this thread is lower
                # priority, so the first match seen wins its slot; higher-
                # priority threads already in nlist may still replace it.
                matched = thread.caps
                stop = True
                break
        clist = nlist
        pos += 1
        if stop and not clist:
            break
        if matched is not None and not clist:
            break
    return matched


# ------------------------------------------------------------ public API ----


class Match:
    """A match result: group spans, mirroring ``re.Match``."""

    __slots__ = ("_text", "_spans", "_pattern", "_pos", "_names")

    def __init__(self, text, spans, pattern, pos, names=None):
        self._text = text
        self._spans = spans
        self._pattern = pattern
        self._pos = pos
        self._names = names or {}

    @property
    def string(self) -> str:
        return self._text

    @property
    def pos(self) -> int:
        return self._pos

    def _span(self, n) -> Tuple[int, int]:
        if isinstance(n, str):
            if n not in self._names:
                raise IndexError(f"no such group: {n!r}")
            n = self._names[n] + 1  # groupindex is 0-based; spans[0] is match
        if not isinstance(n, int) or n < 0 or n >= len(self._spans):
            raise IndexError("no such group")
        return self._spans[n]

    def span(self, n: int = 0) -> Tuple[int, int]:
        return self._span(n)

    def start(self, n: int = 0) -> int:
        return self._span(n)[0]

    def end(self, n: int = 0) -> int:
        return self._span(n)[1]

    def group(self, *ids: int):
        if not ids:
            ids = (0,)
        if len(ids) == 1:
            s, e = self._span(ids[0])
            return None if s < 0 else self._text[s:e]
        return tuple(
            (None if self._span(i)[0] < 0 else self._text[self._span(i)[0] : self._span(i)[1]])
            for i in ids
        )

    def groups(self, default=None):
        out = []
        for s, e in self._spans[1:]:
            out.append(default if s < 0 else self._text[s:e])
        return tuple(out)

    def groupdict(self, default=None):
        """Mapping of named groups to their matched text (or ``default``)."""
        out = {}
        for name, idx in self._names.items():
            s, e = self._spans[idx + 1]
            out[name] = default if s < 0 else self._text[s:e]
        return out

    def __getitem__(self, n: int):
        return self.group(n)

    def __repr__(self) -> str:
        s, e = self._spans[0]
        if s < 0:
            return f"<{type(self).__name__} no match>"
        return f"<{type(self).__name__} span=({s}, {e}) match={self._text[s:e]!r}>"


class Regex:
    """A compiled regular expression (created via :func:`compile`)."""

    def __init__(self, pattern: str, flags: int = 0) -> None:
        if not isinstance(pattern, str):
            raise TypeError("pattern must be a string")
        self.pattern = pattern
        self.flags = flags
        parser = _Parser(pattern, flags)
        ast, ngroups = parser.parse()
        self._program = _build(
            ast,
            ngroups,
            bool(flags & IGNORECASE),
            bool(flags & MULTILINE),
            pattern,
        )
        self._ngroups = ngroups
        self.groupindex = dict(parser.group_names)  # name -> 0-based group index

    @property
    def groups(self) -> int:
        return self._ngroups

    def _run(self, text: str, pos: int, mode: str) -> Optional[Match]:
        caps = _simulate(self._program, text, pos, mode)
        if caps is None:
            return None
        spans = []
        for i in range(self._ngroups + 1):
            spans.append((caps[2 * i], caps[2 * i + 1]))
        return Match(text, spans, self.pattern, pos, self.groupindex)

    def search(self, text: str, pos: int = 0) -> Optional[Match]:
        """Scan forward for the leftmost-first match."""
        return self._run(text, pos, "search")

    def match(self, text: str, pos: int = 0) -> Optional[Match]:
        """Match only at ``pos`` (like Python's ``re.match``)."""
        return self._run(text, pos, "match")

    def fullmatch(self, text: str, pos: int = 0) -> Optional[Match]:
        """Match only if the whole string (from ``pos``) is consumed."""
        return self._run(text, pos, "full")

    def finditer(self, text: str, pos: int = 0) -> Iterator[Match]:
        """Yield non-overlapping matches left to right."""
        n = len(text)
        p = pos
        while p <= n:
            m = self.search(text, p)
            if m is None:
                return
            yield m
            s, e = m.span()
            p = e if e > s else s + 1

    def findall(self, text: str, pos: int = 0) -> List[Any]:
        """Return strings (0 groups), group strings (1 group) or tuples."""
        out: List[Any] = []
        for m in self.finditer(text, pos):
            if self._ngroups == 0:
                out.append(m.group(0))
            elif self._ngroups == 1:
                out.append(m.group(1) or "")
            else:
                out.append(tuple(g or "" for g in m.groups()))
        return out

    def sub(self, repl: str, text: str, count: int = 0) -> str:
        """Replace up to ``count`` (0 = all) non-overlapping matches."""
        if not isinstance(repl, str):
            raise TypeError("repl must be a string")
        out: List[str] = []
        n = len(text)
        p = 0
        done = 0
        while p <= n and (count == 0 or done < count):
            m = self.search(text, p)
            if m is None:
                break
            s, e = m.span()
            out.append(text[p:s])
            out.append(_expand(repl, m))
            done += 1
            p = e if e > s else s + 1
        out.append(text[p:])
        return "".join(out)

    def __repr__(self) -> str:
        return f"<regex_engine.Regex pattern={self.pattern!r} flags={self.flags}>"


def _expand(repl: str, m: Match) -> str:
    out: List[str] = []
    i = 0
    n = len(repl)
    while i < n:
        c = repl[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            raise ValueError("bad escape: trailing backslash in replacement")
        nxt = repl[i + 1]
        if nxt.isdigit():
            g = int(nxt)
            i += 2
        elif nxt == "g":
            closer = None
            if i + 2 < n and repl[i + 2] in "<{":
                closer = ">" if repl[i + 2] == "<" else "}"
            if closer is None:
                raise ValueError("bad escape: expected \\g<...> in replacement")
            j = repl.find(closer, i + 3)
            if j < 0:
                raise ValueError(f"bad escape: unterminated \\g{repl[i + 2]} in replacement")
            try:
                g = int(repl[i + 3 : j])
            except ValueError:
                raise ValueError("bad group reference in replacement") from None
            i = j + 1
        elif nxt == "\\":
            out.append("\\")
            i += 2
            continue
        elif nxt in "nrt":
            out.append({"n": "\n", "r": "\r", "t": "\t"}[nxt])
            i += 2
            continue
        else:
            raise ValueError(f"bad escape \\{nxt} in replacement")
        try:
            val = m.group(g)
        except IndexError:
            raise ValueError(f"invalid group reference {g}") from None
        out.append("" if val is None else val)
    return "".join(out)


def compile(pattern: str, flags: int = 0) -> Regex:  # noqa: A001 - mirrors re.compile
    """Compile ``pattern`` into a :class:`Regex`."""
    return Regex(pattern, flags)


def search(pattern: str, text: str, flags: int = 0) -> Optional[Match]:
    return Regex(pattern, flags).search(text)


def match(pattern: str, text: str, flags: int = 0) -> Optional[Match]:
    return Regex(pattern, flags).match(text)


def fullmatch(pattern: str, text: str, flags: int = 0) -> Optional[Match]:
    return Regex(pattern, flags).fullmatch(text)


def findall(pattern: str, text: str, flags: int = 0) -> List[Any]:
    return Regex(pattern, flags).findall(text)


def finditer(pattern: str, text: str, flags: int = 0) -> Iterator[Match]:
    return Regex(pattern, flags).finditer(text)


def sub(pattern: str, repl: str, text: str, count: int = 0, flags: int = 0) -> str:
    return Regex(pattern, flags).sub(repl, text, count)
