"""Tests for regex-engine: differential corpus vs ``re`` plus engine internals.

The corpus is curated (dozens of hand-picked cases, never random) and asserts
agreement with Python's ``re`` on spans, groups, findall and sub results.
"""

import os
import re as pyre
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import regex_engine as rx
from regex_engine import ParseError, compile as rx_compile

# ---------------------------------------------------------------- corpora ---
# Curated (pattern, text, flags) cases. Agreement with ``re`` is asserted.

SEARCH_CASES = [
    ("abc", "xxabcyy", 0),
    ("abc", "nope", 0),
    ("a.c", "abc", 0),
    ("a.c", "a\nc", 0),
    ("a.c", "a\nc", rx.DOTALL),
    ("\\d+", "abc123def", 0),
    ("\\w+\\s\\w+", "hello   world", 0),
    ("\\s+", "a \t\nb", 0),
    ("[a-z]+", "AbcDef", 0),
    ("[^0-9]+", "123abc456", 0),
    ("[abc]", "cba", 0),
    ("[a-cx-z]", "qxcq", 0),
    ("[]]", "]", 0),
    ("[-a]", "-", 0),
    ("[.]", ".", 0),
    ("a*", "aaa", 0),
    ("a*", "", 0),
    ("a+", "aaa", 0),
    ("a?", "b", 0),
    ("a{2,3}", "aaaa", 0),
    ("a{2}", "aaa", 0),
    ("a{2,}", "aaaaa", 0),
    ("a{0}b", "b", 0),
    ("a+?", "aaa", 0),
    ("a*?b", "aaab", 0),
    ("a.*?b", "axxbxb", 0),
    ("(a+)(a+)", "aaa", 0),
    ("(a+?)(a+)", "aaa", 0),
    ("cat|dog", "ilovedog", 0),
    ("a|ab", "ab", 0),
    ("(a|ab)c", "abc", 0),
    ("(a(b)c)", "abc", 0),
    ("((a)(b))c", "abc", 0),
    ("(a+)(b+)?", "aab", 0),
    ("^abc", "abc", 0),
    ("^abc", "xabc", 0),
    ("abc$", "abc\n", 0),
    ("abc$", "abcx", 0),
    ("^a$", "a\nb", rx.MULTILINE),
    ("^b$", "a\nb", rx.MULTILINE),
    ("^b$", "a\nb", 0),
    ("$", "abc", 0),
    ("^", "abc", 0),
    ("^", "", 0),
    ("^$", "abc", 0),
    ("^a", "ba", 0),
    ("[A-Za-z_][A-Za-z0-9_]*", "99foo_bar baz", 0),
    ("(\\d+)-(\\d+)", "12-34", 0),
    ("(foo|bar|baz)+", "barbazfoo", 0),
    ("a.c|b.d", "xxbzd", 0),
    ("^(?:abc)+$", "abcabc", 0),
    ("^(?:abc)+$", "abcabd", 0),
    ("(a|b)*c", "ababac", 0),
    ("colou?r", "color", 0),
    ("colou?r", "colour", 0),
    ("colou?r", "colouur", 0),
    ("()", "x", 0),
    ("a|", "b", 0),
    (".*", "", 0),
    ("a.*b", "axxb", 0),
    ("^.*$", "abc", 0),
    ("\\w+@\\w+\\.\\w+", "mail: bob@example.com", 0),
    ("(a+)+b", "a" * 24, 0),
    ("(a+)+b", "a" * 24 + "b", 0),
    ("café", "un café frais", 0),
    ("[àé]", "xé", 0),
    ("\\d{3}", "12 4567", 0),
    ("(cat|dog|fish)+", "dogfishcat", 0),
    ("end$", "the end\n", 0),
    ("^\\s*#", "  # comment", 0),
    ("\\t", "a\tb", 0),
    ("\\n", "a\nb", 0),
    ("\\\\", "a\\b", 0),
    ("\\[", "[x", 0),
    ("\\.", "a.b", 0),
    ("\\.", "axb", 0),
    ("x?y*z", "yyz", 0),
    ("\\+", "a+b", 0),
    ("\\d+", "12", rx.IGNORECASE),
    ("HELLO", "say hello", rx.IGNORECASE),
    ("[A-Z]+", "abcDEF", rx.IGNORECASE),
    ("[^A-Z]+", "abcDEF", rx.IGNORECASE),
    ("a.b", "a\nb", rx.DOTALL),
    ("^b", "a\nb", rx.MULTILINE),
    ("foo$", "foo\nbar", rx.MULTILINE),
    ("(a)(b)(c)", "abc", 0),
    ("x*", "y", 0),
    ("a.c", "a-c", 0),
    ("\\w+", "héllo", 0),
    ("\\W", " ", 0),
    ("\\S", " a", 0),
    ("oo", "food", 0),
    ("(a+)(b+)(c+)", "aabbcc", 0),
    ("a{1,2}?", "aa", 0),
    ("(foo)?bar", "bar", 0),
    ("^a|^b", "b", 0),
    ("a$|b$", "b", 0),
    ("[0-9]{2,4}", "123456", 0),
    ("\\.(?:exe|dll)$", "setup.dll", 0),
]

FINDALL_CASES = [
    ("\\d+", "a1b22c333", 0),
    ("[aeiou]", "hello world", 0),
    ("(\\d)(\\w)", "1a 2b", 0),
    ("a*", "aab", 0),
    ("x*", "axb", 0),
    ("(cat|dog)", "cat dog cat", 0),
    ("^\\w+", "one two", rx.MULTILINE),
    ("(a)(b)?", "ab a", 0),
    ("\\s+", "a  b", 0),
    ("x+", "xxaxx", 0),
]

SUB_CASES = [
    ("\\d+", "#", "a1b22c333", 0),
    ("(a)(b)", r"\2\1", "ab", 0),
    ("\\s+", " ", "a   b", 0),
    ("cat", "dog", "cat cat cat", 1),
    ("a", "X", "aaa", 0),
    ("b", "", "abc", 0),
    ("\\w+", r"[\g<0>]", "hi there", 0),
    ("x", r"\\", "x", 0),
]

FULL_CASES = [
    ("a+", "aaa", 0),
    ("a+", "aaab", 0),
    ("a", "a\n", 0),
    ("a$", "a\n", 0),
    ("(a|b)+", "abab", 0),
    ("x", "", 0),
    ("", "abc", 0),
]

MATCH_CASES = [
    ("a+", "aaa", 0),
    ("a+", "baa", 0),
    ("^a", "a\nb", rx.MULTILINE),
]


class DifferentialSearchTests(unittest.TestCase):
    def test_search_corpus_agrees_with_python_re(self):
        self.assertGreaterEqual(len(SEARCH_CASES), 60)
        for pat, text, flags in SEARCH_CASES:
            with self.subTest(pattern=pat, text=text, flags=flags):
                expected = pyre.search(pat, text, flags)
                exp_span = expected.span() if expected else None
                exp_groups = expected.groups() if expected else None
                got = rx.search(pat, text, flags)
                got_span = got.span() if got else None
                got_groups = got.groups() if got else None
                self.assertEqual(got_span, exp_span)
                self.assertEqual(got_groups, exp_groups)

    def test_findall_corpus_agrees_with_python_re(self):
        for pat, text, flags in FINDALL_CASES:
            with self.subTest(pattern=pat, text=text, flags=flags):
                self.assertEqual(rx.findall(pat, text, flags), pyre.findall(pat, text, flags))

    def test_sub_corpus_agrees_with_python_re(self):
        for pat, repl, text, flags in SUB_CASES:
            with self.subTest(pattern=pat, repl=repl, text=text):
                self.assertEqual(
                    rx.sub(pat, repl, text, flags=flags),
                    pyre.sub(pat, repl, text, flags=flags),
                )

    def test_fullmatch_and_match_agree_with_python_re(self):
        for pat, text, flags in FULL_CASES:
            with self.subTest(mode="fullmatch", pattern=pat, text=text):
                expected = pyre.fullmatch(pat, text, flags)
                got = rx.fullmatch(pat, text, flags)
                self.assertEqual(got.span() if got else None,
                                 expected.span() if expected else None)
        for pat, text, flags in MATCH_CASES:
            with self.subTest(mode="match", pattern=pat, text=text):
                expected = pyre.match(pat, text, flags)
                got = rx.match(pat, text, flags)
                self.assertEqual(got.span() if got else None,
                                 expected.span() if expected else None)

    def test_flag_constants_match_re_module(self):
        self.assertEqual(rx.IGNORECASE, pyre.IGNORECASE)
        self.assertEqual(rx.MULTILINE, pyre.MULTILINE)
        self.assertEqual(rx.DOTALL, pyre.DOTALL)


class EngineBehaviourTests(unittest.TestCase):
    def test_capture_groups_nested_and_unmatched(self):
        m = rx.search(r"((a)(b))c", "abc")
        self.assertEqual(m.span(), (0, 3))
        self.assertEqual(m.groups(), ("ab", "a", "b"))
        m = rx.search(r"(a+)(b+)?", "aab")
        self.assertEqual(m.groups(), ("aa", "b"))
        m = rx.search(r"(foo)?bar", "bar")
        self.assertIsNone(m.group(1))
        self.assertEqual(m.group(0), "bar")
        m = rx.search(r"(\d+)-(\d+)", "ext 123-456")
        self.assertEqual(m.span(1), (4, 7))
        self.assertEqual(m.span(2), (8, 11))
        self.assertEqual(m[2], "456")
        m = rx.search(r"(?P<word>\w+)-(?P<num>\d+)", "abc-42")
        self.assertEqual(m.group("word"), "abc")
        self.assertEqual(m.group("num"), "42")
        self.assertEqual(m.groupdict(), {"word": "abc", "num": "42"})
        m = rx.search(r"(?P<a>x)?y", "y")
        self.assertEqual(m.groupdict(), {"a": None})

    def test_greedy_and_lazy_preference(self):
        self.assertEqual(rx.search(r"a*", "aaa").span(), (0, 3))
        self.assertEqual(rx.search(r"a+?", "aaa").span(), (0, 1))
        self.assertEqual(rx.search(r"a.*?b", "axxbxb").span(), (0, 4))
        self.assertEqual(rx.search(r"a.*b", "axxbxb").span(), (0, 6))
        self.assertEqual(rx.search(r"(a+)(a+)", "aaa").groups(), ("aa", "a"))
        self.assertEqual(rx.search(r"(a+?)(a+)", "aaa").groups(), ("a", "aa"))
        self.assertEqual(rx.search(r"a{2,3}", "aaaa").span(), (0, 3))

    def test_catastrophic_backtracking_completes_quickly(self):
        # ``(a+)+b`` is exponential for backtracking engines; the NFA
        # simulation must finish in linear time (generous 5s bound).
        started = time.perf_counter()
        self.assertIsNone(rx.search(r"(a+)+b", "a" * 30))
        m = rx.search(r"(a+)+b", "a" * 30 + "b")
        elapsed = time.perf_counter() - started
        self.assertIsNotNone(m)
        self.assertEqual(m.span(), (0, 31))
        self.assertLess(elapsed, 5.0)

    def test_anchors_and_multiline_semantics(self):
        # Without MULTILINE: ^ only at string start, $ at end or before a
        # trailing newline (Python semantics).
        self.assertIsNotNone(rx.search(r"^abc", "abc"))
        self.assertIsNone(rx.search(r"^abc", "xabc"))
        self.assertIsNotNone(rx.search(r"abc$", "abc\n"))
        self.assertEqual(rx.search(r"abc$", "abc\n").span(), (0, 3))
        self.assertIsNone(rx.search(r"^b$", "a\nb"))
        # With MULTILINE: anchors match at every line boundary.
        m = rx.search(r"^b$", "a\nb", rx.MULTILINE)
        self.assertEqual(m.span(), (2, 3))
        m = rx.search(r"^a$", "a\nb", rx.MULTILINE)
        self.assertEqual(m.span(), (0, 1))
        self.assertEqual(rx.search(r"$", "abc").span(), (3, 3))
        self.assertEqual(rx.search(r"^", "").span(), (0, 0))

    def test_character_classes_and_escapes(self):
        self.assertEqual(rx.search(r"[a-z]+", "AbcDef").span(), (1, 3))
        self.assertEqual(rx.search(r"[^0-9]+", "123abc456").span(), (3, 6))
        self.assertEqual(rx.search(r"[]]", "x]y").span(), (1, 2))
        self.assertEqual(rx.search(r"[-a]", "x-y").span(), (1, 2))
        self.assertEqual(rx.search(r"\d+", "abc123").group(), "123")
        self.assertEqual(rx.search(r"\w+", "héllo").group(), "héllo")
        self.assertEqual(rx.search(r"\s+", "a \tb").group(), " \t")
        self.assertEqual(rx.search(r"\\", "a\\b").group(), "\\")
        self.assertEqual(rx.search(r"\.", "a.b").group(), ".")
        self.assertIsNone(rx.search(r"\.", "axb"))
        # DOTALL lets '.' cross newlines; default does not.
        self.assertIsNone(rx.search(r"a.c", "a\nc"))
        self.assertIsNotNone(rx.search(r"a.c", "a\nc", rx.DOTALL))

    def test_invalid_patterns_report_position(self):
        cases = [
            ("(abc", 0, "missing )"),
            ("*a", 0, "nothing to repeat"),
            ("a**", 2, "multiple repeat"),
            ("[abc", 0, "unterminated character set"),
            ("a)b", 1, "unbalanced parenthesis"),
            ("a{1,0}", 1, "min repeat greater than max"),
            ("\\b", 0, "not supported"),
            ("a\\", 1, "incomplete escape"),
            ("(?=x)", 0, "lookahead"),
            ("(?P<x>a)(?P<x>b)", 8, "duplicate group name"),
            ("\\q", 0, "bad escape"),
        ]
        for pattern, pos, fragment in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaises(ParseError) as cm:
                    rx_compile(pattern)
                self.assertEqual(cm.exception.pos, pos)
                self.assertIn(fragment, str(cm.exception))
                self.assertIn(f"position {pos}", str(cm.exception))

    def test_match_object_api(self):
        m = rx.search(r"(\d+)-(\d+)", "id: 12-345")
        self.assertEqual(m.string, "id: 12-345")
        self.assertEqual(m.pos, 0)
        self.assertEqual(m.span(), (4, 10))
        self.assertEqual(m.start(), 4)
        self.assertEqual(m.end(), 10)
        self.assertEqual(m.group(), "12-345")
        self.assertEqual(m.group(1), "12")
        self.assertEqual(m.group(2), "345")
        self.assertEqual(m.group(1, 2), ("12", "345"))
        self.assertEqual(m.span(1), (4, 6))
        self.assertEqual(m.span(2), (7, 10))
        self.assertIn("span=(4, 10)", repr(m))

    def test_finditer_non_overlapping_and_empty_advance(self):
        spans = [m.span() for m in rx.finditer(r"\d+", "a1b22c333")]
        self.assertEqual(spans, [(1, 2), (3, 5), (6, 9)])
        # Empty matches advance by one position without looping forever.
        spans = [m.span() for m in rx.finditer(r"a*", "aab")]
        self.assertEqual(spans, [(0, 2), (2, 2), (3, 3)])
        spans = [m.span() for m in rx.finditer(r"x*", "axb")]
        self.assertEqual(spans, [(0, 0), (1, 2), (2, 2), (3, 3)])

    def test_sub_replacement_forms_and_count(self):
        self.assertEqual(rx.sub(r"\d+", "#", "a1b22c333"), "a#b#c#")
        self.assertEqual(rx.sub(r"(a)(b)", r"\2\1", "ab"), "ba")
        self.assertEqual(rx.sub(r"cat", "dog", "cat cat cat", count=1), "dog cat cat")
        self.assertEqual(rx.sub(r"b", "", "abc"), "ac")
        with self.assertRaises(ValueError):
            rx.sub(r"a", r"\q", "a")
        with self.assertRaises(ValueError):
            rx.sub(r"a", r"\9", "a")

    def test_compile_returns_regex_and_repr(self):
        prog = rx_compile(r"a+", rx.IGNORECASE)
        self.assertIsInstance(prog, rx.Regex)
        self.assertEqual(prog.groups, 0)
        self.assertEqual(prog.pattern, "a+")
        self.assertEqual(prog.flags, rx.IGNORECASE)
        self.assertIn("a+", repr(prog))


if __name__ == "__main__":
    unittest.main()
