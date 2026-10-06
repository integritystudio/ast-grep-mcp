"""Regression tests for small robustness fixes: temp cleanup, regex escaping, stdin input."""

import subprocess
import tempfile

from ast_grep_mcp.features.complexity.metrics import count_pattern_matches
from ast_grep_mcp.features.deduplication.coverage import CoverageDetector
from ast_grep_mcp.utils import templates


def test_java_format_temp_file_removed_on_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(templates.shutil, "which", lambda _: "/usr/bin/google-java-format")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="google-java-format", timeout=1)

    monkeypatch.setattr(templates.subprocess, "run", timeout)

    assert templates._try_google_java_format("class A {}") is None
    assert list(tmp_path.iterdir()) == []


def test_coverage_handles_regex_metacharacters_in_file_name() -> None:
    detector = CoverageDetector()
    # An unbalanced "[" made re.compile raise instead of matching
    content = "from notes[draft import helper\n"
    assert detector._check_test_file_references_source("test_x.py", "src/notes[draft.py", "python", content=content)


def test_coverage_fallback_requires_whole_word() -> None:
    detector = CoverageDetector()
    assert not detector._check_test_file_references_source("test_x.js", "src/index.js", "ruby", content="reindex()\n")


def test_count_pattern_matches_reads_code_argument() -> None:
    assert count_pattern_matches("print(1)\nprint(2)\n", "print($A)", "python") == 2
