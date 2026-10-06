"""Tests that distinct ast-grep queries never share a cache key."""

from ast_grep_mcp.core.cache import QueryCache

PROJECT = "/project"


def _key(args):
    return QueryCache(max_size=1, ttl_seconds=60)._make_key("run", args, PROJECT)


def test_swapped_flag_values_get_distinct_keys() -> None:
    a = ["--pattern", "print($A)", "--selector", "call", "--json=stream"]
    b = ["--pattern", "call", "--selector", "print($A)", "--json=stream"]
    assert _key(a) != _key(b)


def test_separator_inside_value_does_not_collide() -> None:
    assert _key(["--pattern", "a|b"]) != _key(["--pattern", "a", "b"])


def test_identical_args_share_a_key() -> None:
    args = ["--pattern", "x", "--lang", "python"]
    assert _key(args) == _key(list(args))
