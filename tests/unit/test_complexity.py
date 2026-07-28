"""Unit tests for complexity analysis functions."""

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from ast_grep_mcp.utils.console_logger import console

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ast_grep_mcp.features.complexity.analyzer import (
    analyze_file_complexity,
    calculate_cognitive_complexity,
    calculate_cyclomatic_complexity,
)
from ast_grep_mcp.features.complexity.metrics import (
    calculate_nesting_depth,
    get_complexity_patterns,
    strip_non_code,
)
from ast_grep_mcp.features.complexity.storage import ComplexityStorage
from ast_grep_mcp.models.complexity import ComplexityMetrics, ComplexityThresholds, FunctionComplexity


class TestCyclomaticComplexity:
    """Test cyclomatic complexity calculation."""

    def test_simple_function(self):
        """Simple function with no branches should have complexity 1."""
        code = """
def simple():
    return 42
"""
        result = calculate_cyclomatic_complexity(code, "python")
        assert result == 1

    def test_single_if(self):
        """Single if statement adds 1 to complexity."""
        code = """
def func(x):
    if x > 0:
        return x
    return 0
"""
        result = calculate_cyclomatic_complexity(code, "python")
        assert result == 2

    def test_if_elif_else(self):
        """if-elif chain counts each branch."""
        code = """
def func(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
"""
        result = calculate_cyclomatic_complexity(code, "python")
        assert result >= 3  # 1 + if + elif (may count else: too)

    def test_for_loop(self):
        """For loop adds 1 to complexity."""
        code = """
def func(items):
    for item in items:
        console.log(item)
"""
        result = calculate_cyclomatic_complexity(code, "python")
        assert result == 2

    def test_while_loop(self):
        """While loop adds 1 to complexity."""
        code = """
def func(n):
    while n > 0:
        n -= 1
"""
        result = calculate_cyclomatic_complexity(code, "python")
        assert result == 2

    def test_logical_operators(self):
        """Logical operators add to complexity."""
        code = """
def func(x, y):
    if x > 0 and y > 0:
        return True
    if x < 0 or y < 0:
        return False
"""
        result = calculate_cyclomatic_complexity(code, "python")
        assert result == 5  # 1 + 2 ifs + and + or

    def test_nested_branches(self):
        """Nested branches accumulate complexity."""
        code = """
def func(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x
    return 0
"""
        result = calculate_cyclomatic_complexity(code, "python")
        assert result == 3  # 1 + 2 ifs

    def test_exception_handling(self):
        """Exception handlers add complexity."""
        code = """
def func():
    try:
        risky()
    except ValueError:
        handle_value()
    except TypeError:
        handle_type()
"""
        result = calculate_cyclomatic_complexity(code, "python")
        assert result == 3  # 1 + 2 excepts

    def test_with_statement(self):
        """With statement adds complexity."""
        code = """
def func():
    with open('file') as f:
        return f.read()
"""
        result = calculate_cyclomatic_complexity(code, "python")
        assert result == 2

    def test_complex_function(self):
        """Complex function with multiple branches."""
        code = """
def complex_func(data):
    result = []
    for item in data:
        if item > 0:
            if item % 2 == 0:
                result.append(item * 2)
            else:
                result.append(item)
        elif item < 0:
            result.append(-item)
    return result
"""
        result = calculate_cyclomatic_complexity(code, "python")
        assert result >= 5

    def test_typescript_if(self):
        """TypeScript if statement."""
        code = """
function func(x: number): number {
    if (x > 0) {
        return x;
    }
    return 0;
}
"""
        result = calculate_cyclomatic_complexity(code, "typescript")
        assert result == 2

    def test_typescript_logical_operators(self):
        """TypeScript logical operators."""
        code = """
function func(x: number, y: number): boolean {
    return x > 0 && y > 0 || x < 0;
}
"""
        result = calculate_cyclomatic_complexity(code, "typescript")
        assert result == 3  # 1 + && + ||

    def test_javascript_switch(self):
        """JavaScript switch statement."""
        code = """
function func(x) {
    switch (x) {
        case 1:
            return 'one';
        case 2:
            return 'two';
        default:
            return 'other';
    }
}
"""
        result = calculate_cyclomatic_complexity(code, "javascript")
        assert result >= 3  # 1 + switch + 2 cases

    def test_java_for_loop(self):
        """Java for loop."""
        code = """
public int sum(int[] arr) {
    int total = 0;
    for (int i = 0; i < arr.length; i++) {
        total += arr[i];
    }
    return total;
}
"""
        result = calculate_cyclomatic_complexity(code, "java")
        assert result == 2


class TestCognitiveComplexity:
    """Test cognitive complexity calculation."""

    def test_simple_function(self):
        """Simple function with no branches should have complexity 0."""
        code = """
def simple():
    return 42
"""
        result = calculate_cognitive_complexity(code, "python")
        assert result == 0

    def test_single_if(self):
        """Single if statement adds 1."""
        code = """
def func(x):
    if x > 0:
        return x
    return 0
"""
        result = calculate_cognitive_complexity(code, "python")
        assert result >= 1

    def test_nested_if_penalty(self):
        """Nested if adds nesting penalty."""
        code = """
def func(x, y):
    if x > 0:
        if y > 0:
            return x + y
    return 0
"""
        result = calculate_cognitive_complexity(code, "python")
        # Second if should have nesting penalty
        assert result >= 3

    def test_deeply_nested(self):
        """Deep nesting increases cognitive load significantly."""
        code = """
def func(a, b, c, d):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    return True
    return False
"""
        result = calculate_cognitive_complexity(code, "python")
        # Should be: 1 + (1+1) + (1+2) + (1+3) = 10
        assert result >= 6

    def test_logical_operators_add(self):
        """Logical operators add to cognitive complexity."""
        code = """
def func(x, y):
    if x > 0 and y > 0:
        return True
"""
        result = calculate_cognitive_complexity(code, "python")
        assert result >= 2


class TestNestingDepth:
    """Test nesting depth calculation."""

    def test_no_nesting(self):
        """Function with no nesting should have minimal depth."""
        code = """
def simple():
    return 42
"""
        result = calculate_nesting_depth(code, "python")
        assert result <= 1  # Function body counts as base level

    def test_single_level(self):
        """Single level of nesting."""
        code = """
def func(x):
    if x > 0:
        return x
    return 0
"""
        result = calculate_nesting_depth(code, "python")
        assert result >= 1

    def test_two_levels(self):
        """Two levels of nesting."""
        code = """
def func(x, y):
    if x > 0:
        if y > 0:
            return x + y
    return 0
"""
        result = calculate_nesting_depth(code, "python")
        assert result >= 2

    def test_deep_nesting(self):
        """Deep nesting of 4 levels."""
        code = """
def func(a, b, c, d):
    if a:
        if b:
            if c:
                if d:
                    return True
    return False
"""
        result = calculate_nesting_depth(code, "python")
        assert result >= 4

    def test_loop_nesting(self):
        """Loops contribute to nesting."""
        code = """
def func(matrix):
    for row in matrix:
        for cell in row:
            console.log(cell)
"""
        result = calculate_nesting_depth(code, "python")
        assert result >= 2


class TestComplexityPatterns:
    """Test language pattern retrieval."""

    def test_python_patterns(self):
        """Get Python patterns."""
        patterns = get_complexity_patterns("python")
        assert "function" in patterns
        assert "branches" in patterns
        assert patterns["function"] == "def $NAME($$$)"

    def test_typescript_patterns(self):
        """Get TypeScript patterns."""
        patterns = get_complexity_patterns("typescript")
        assert "function" in patterns
        assert "arrow_function" in patterns

    def test_javascript_patterns(self):
        """Get JavaScript patterns."""
        patterns = get_complexity_patterns("javascript")
        assert "function" in patterns
        assert patterns["function"] == "function $NAME($$$) { $$$ }"

    def test_java_patterns(self):
        """Get Java patterns."""
        patterns = get_complexity_patterns("java")
        assert "function" in patterns
        assert "$TYPE" in patterns["function"]

    def test_unknown_language_defaults_to_python(self):
        """Unknown language defaults to Python patterns."""
        patterns = get_complexity_patterns("unknown")
        assert patterns == get_complexity_patterns("python")

    def test_case_insensitive(self):
        """Language names are case insensitive."""
        assert get_complexity_patterns("Python") == get_complexity_patterns("python")
        assert get_complexity_patterns("TYPESCRIPT") == get_complexity_patterns("typescript")


class TestComplexityDataClasses:
    """Test complexity data classes."""

    def test_complexity_metrics(self):
        """Test ComplexityMetrics dataclass."""
        metrics = ComplexityMetrics(cyclomatic=5, cognitive=10, nesting_depth=3, lines=25, parameter_count=2)
        assert metrics.cyclomatic == 5
        assert metrics.cognitive == 10
        assert metrics.nesting_depth == 3
        assert metrics.lines == 25
        assert metrics.parameter_count == 2

    def test_complexity_metrics_defaults(self):
        """Test ComplexityMetrics default values."""
        metrics = ComplexityMetrics(cyclomatic=1, cognitive=0, nesting_depth=0, lines=5)
        assert metrics.parameter_count == 0

    def test_function_complexity(self):
        """Test FunctionComplexity dataclass."""
        metrics = ComplexityMetrics(5, 10, 3, 25, 2)
        func = FunctionComplexity(
            file_path="/path/to/file.py",
            function_name="my_func",
            start_line=10,
            end_line=35,
            metrics=metrics,
            language="python",
            exceeds=["cyclomatic", "nesting"],
        )
        assert func.file_path == "/path/to/file.py"
        assert func.function_name == "my_func"
        assert func.exceeds == ["cyclomatic", "nesting"]

    def test_complexity_thresholds(self):
        """Test ComplexityThresholds dataclass."""
        thresholds = ComplexityThresholds()
        assert thresholds.cyclomatic == 10
        assert thresholds.cognitive == 15
        assert thresholds.nesting_depth == 4
        assert thresholds.lines == 50

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = ComplexityThresholds(cyclomatic=5, cognitive=8, nesting_depth=3, lines=30)
        assert thresholds.cyclomatic == 5
        assert thresholds.cognitive == 8


class TestComplexityStorage:
    """Test SQLite storage for complexity results."""

    def test_storage_initialization(self):
        """Test storage initializes database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ComplexityStorage(db_path)
            assert db_path.exists()
            assert storage.db_path == db_path

    def test_get_or_create_project(self):
        """Test project creation and retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ComplexityStorage(db_path)

            # Create project
            project_id = storage.get_or_create_project("/path/to/project")
            assert project_id > 0

            # Get same project
            same_id = storage.get_or_create_project("/path/to/project")
            assert same_id == project_id

    def test_store_analysis_run(self):
        """Test storing analysis run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ComplexityStorage(db_path)

            results = {
                "total_functions": 10,
                "total_files": 3,
                "avg_cyclomatic": 5.5,
                "avg_cognitive": 8.2,
                "max_cyclomatic": 15,
                "max_cognitive": 20,
                "max_nesting": 4,
                "violation_count": 2,
                "duration_ms": 1500,
            }

            functions = [
                FunctionComplexity(
                    file_path="/path/file.py",
                    function_name="func1",
                    start_line=1,
                    end_line=20,
                    metrics=ComplexityMetrics(5, 8, 2, 20, 2),
                    language="python",
                    exceeds=[],
                )
            ]

            run_id = storage.store_analysis_run("/path/to/project", results, functions, "abc123", "main")
            assert run_id > 0

    def test_get_project_trends(self):
        """Test retrieving project trends."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = ComplexityStorage(db_path)

            # Store a run
            results = {
                "total_functions": 10,
                "total_files": 3,
                "avg_cyclomatic": 5.5,
                "avg_cognitive": 8.2,
                "max_cyclomatic": 15,
                "max_cognitive": 20,
                "max_nesting": 4,
                "violation_count": 2,
                "duration_ms": 1500,
            }
            storage.store_analysis_run("/path/to/project", results, [])

            # Get trends
            trends = storage.get_project_trends("/path/to/project", days=30)
            assert len(trends) == 1
            assert trends[0]["total_functions"] == 10


class TestAnalyzeFileComplexity:
    """Test file complexity analysis."""

    def test_analyze_empty_file(self):
        """Test analyzing file with no functions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# Just a comment\n")
            f.flush()

            try:
                thresholds = ComplexityThresholds()
                results = analyze_file_complexity(f.name, "python", thresholds)
                # May or may not find functions depending on ast-grep behavior
                assert isinstance(results, list)
            finally:
                os.unlink(f.name)

    def test_analyze_simple_function(self):
        """Test analyzing file with simple function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def simple():
    return 42

def with_branch(x):
    if x > 0:
        return x
    return 0
""")
            f.flush()

            try:
                thresholds = ComplexityThresholds()
                results = analyze_file_complexity(f.name, "python", thresholds)
                # Should find functions (depends on ast-grep)
                assert isinstance(results, list)
            finally:
                os.unlink(f.name)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_code(self):
        """Empty code should return minimal complexity."""
        result = calculate_cyclomatic_complexity("", "python")
        assert result == 1

        result = calculate_cognitive_complexity("", "python")
        assert result == 0

        result = calculate_nesting_depth("", "python")
        assert result == 0

    def test_code_with_only_comments(self):
        """Code with only comments."""
        code = """
# This is a comment
# Another comment
"""
        result = calculate_cyclomatic_complexity(code, "python")
        assert result == 1

    def test_multiline_string(self):
        """Multiline strings shouldn't be counted as branches."""
        code = '''
def func():
    """
    if this were code
    for it would count
    while increasing complexity
    """
    return True
'''
        # The docstring contains keywords but they're in a string
        result = calculate_cyclomatic_complexity(code, "python")
        # This is a known limitation - string contents may be counted
        assert result >= 1

    def test_very_long_function(self):
        """Very long function should work correctly."""
        lines = ["def long_func():"]
        for i in range(100):
            lines.append(f"    x{i} = {i}")
        lines.append("    return x99")
        code = "\n".join(lines)

        result = calculate_nesting_depth(code, "python")
        assert result == 1  # Only one level of indentation

    def test_tabs_vs_spaces(self):
        """Handle both tabs and spaces for indentation."""
        code_spaces = """
def func():
    if True:
        return 1
"""
        # Both should work (though tabs may give different depth)
        result_spaces = calculate_nesting_depth(code_spaces, "python")
        assert result_spaces >= 1


class TestBenchmark:
    """Performance benchmark tests for complexity analysis."""

    def _generate_function(self, idx: int, complexity: str = "medium") -> str:
        """Generate a synthetic function with specified complexity.

        Args:
            idx: Function index for unique naming
            complexity: 'simple', 'medium', or 'complex'

        Returns:
            Python function code string
        """
        if complexity == "simple":
            return f"""
def func_{idx}(x):
    return x * 2
"""
        elif complexity == "complex":
            return f"""
def func_{idx}(a, b, c, d):
    result = 0
    for i in range(a):
        if i % 2 == 0:
            for j in range(b):
                if j > c:
                    result += i * j
                elif j < d:
                    result -= i
                else:
                    try:
                        result += i / j
                    except ZeroDivisionError:
                        result = 0
    return result
"""
        else:  # medium
            return f"""
def func_{idx}(x, y):
    if x > 0:
        if y > 0:
            return x + y
        else:
            return x - y
    elif x < 0:
        return -x
    else:
        for i in range(y):
            if i % 2 == 0:
                x += i
        return x
"""

    def test_cyclomatic_1000_functions(self):
        """Benchmark: Calculate cyclomatic complexity for 1000 functions in <10s."""
        # Generate 1000 functions (mix of complexities)
        functions = []
        for i in range(1000):
            if i % 3 == 0:
                functions.append(self._generate_function(i, "simple"))
            elif i % 3 == 1:
                functions.append(self._generate_function(i, "medium"))
            else:
                functions.append(self._generate_function(i, "complex"))

        start_time = time.time()

        for code in functions:
            calculate_cyclomatic_complexity(code, "python")

        elapsed = time.time() - start_time

        assert elapsed < 10.0, f"Cyclomatic complexity for 1000 functions took {elapsed:.2f}s (>10s)"
        console.log(f"\nCyclomatic complexity benchmark: {elapsed:.2f}s for 1000 functions")

    def test_cognitive_1000_functions(self):
        """Benchmark: Calculate cognitive complexity for 1000 functions in <10s."""
        functions = []
        for i in range(1000):
            if i % 3 == 0:
                functions.append(self._generate_function(i, "simple"))
            elif i % 3 == 1:
                functions.append(self._generate_function(i, "medium"))
            else:
                functions.append(self._generate_function(i, "complex"))

        start_time = time.time()

        for code in functions:
            calculate_cognitive_complexity(code, "python")

        elapsed = time.time() - start_time

        assert elapsed < 10.0, f"Cognitive complexity for 1000 functions took {elapsed:.2f}s (>10s)"
        console.log(f"\nCognitive complexity benchmark: {elapsed:.2f}s for 1000 functions")

    def test_nesting_depth_1000_functions(self):
        """Benchmark: Calculate nesting depth for 1000 functions in <10s."""
        functions = []
        for i in range(1000):
            if i % 3 == 0:
                functions.append(self._generate_function(i, "simple"))
            elif i % 3 == 1:
                functions.append(self._generate_function(i, "medium"))
            else:
                functions.append(self._generate_function(i, "complex"))

        start_time = time.time()

        for code in functions:
            calculate_nesting_depth(code, "python")

        elapsed = time.time() - start_time

        assert elapsed < 10.0, f"Nesting depth for 1000 functions took {elapsed:.2f}s (>10s)"
        console.log(f"\nNesting depth benchmark: {elapsed:.2f}s for 1000 functions")

    def test_all_metrics_1000_functions(self):
        """Benchmark: All complexity metrics for 1000 functions in <10s."""
        functions = []
        for i in range(1000):
            if i % 3 == 0:
                functions.append(self._generate_function(i, "simple"))
            elif i % 3 == 1:
                functions.append(self._generate_function(i, "medium"))
            else:
                functions.append(self._generate_function(i, "complex"))

        start_time = time.time()

        for code in functions:
            calculate_cyclomatic_complexity(code, "python")
            calculate_cognitive_complexity(code, "python")
            calculate_nesting_depth(code, "python")

        elapsed = time.time() - start_time

        assert elapsed < 10.0, f"All metrics for 1000 functions took {elapsed:.2f}s (>10s)"
        console.log(f"\nAll metrics benchmark: {elapsed:.2f}s for 1000 functions")

    @pytest.mark.slow
    def test_file_analysis_100_files(self):
        """Benchmark: Analyze 100 files with multiple functions each."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 100 files with 10 functions each
            file_paths = []
            for file_idx in range(100):
                file_path = Path(tmpdir) / f"module_{file_idx}.py"

                content = []
                for func_idx in range(10):
                    global_idx = file_idx * 10 + func_idx
                    if func_idx % 3 == 0:
                        content.append(self._generate_function(global_idx, "simple"))
                    elif func_idx % 3 == 1:
                        content.append(self._generate_function(global_idx, "medium"))
                    else:
                        content.append(self._generate_function(global_idx, "complex"))

                file_path.write_text("\n".join(content))
                file_paths.append(str(file_path))

            start_time = time.time()

            thresholds = ComplexityThresholds()
            total_functions = 0
            for file_path in file_paths:
                results = analyze_file_complexity(file_path, "python", thresholds)
                total_functions += len(results)

            elapsed = time.time() - start_time

            # 100 files * 10 functions = 1000 functions
            # Should complete in reasonable time (<30s)
            assert elapsed < 30.0, f"File analysis for {total_functions} functions took {elapsed:.2f}s (>30s)"
            console.log(f"\nFile analysis benchmark: {elapsed:.2f}s for {total_functions} functions in 100 files")


class TestStripNonCode:
    """Comments and string literals must not be counted as control flow.

    Keyword counting ran over raw source, so prose inflated every metric: a
    docstring using the words "for", "if", "and" or "or" read as decision
    points. utils/futures.py:map_with_per_item_timeout measured cyclomatic 24
    against a body of 9 — 14 of the 23 counted points came from its docstring —
    and tripped the critical-threshold quality gate on that basis alone.
    """

    def test_docstring_keywords_are_not_counted(self):
        documented = '''def f(x):
    """Wait for each item and log it, or skip if the deadline passed.

    Iterate for as long as needed with either strategy.
    """
    return x
'''
        undocumented = "def f(x):\n    return x\n"

        assert calculate_cyclomatic_complexity(documented, "python") == calculate_cyclomatic_complexity(undocumented, "python")

    def test_comment_keywords_are_not_counted(self):
        commented = "def f(x):\n    # loop for a while and check if valid or not\n    return x\n"

        assert calculate_cyclomatic_complexity(commented, "python") == 1

    def test_real_branches_are_still_counted(self):
        code = "def f(x):\n    if x and x > 1:\n        for i in x:\n            pass\n    return x\n"

        # base 1 + if + and + for
        assert calculate_cyclomatic_complexity(code, "python") == 4

    def test_string_literal_keywords_are_not_counted(self):
        code = 'def f():\n    msg = "if this and that or the other"\n    return msg\n'

        assert calculate_cyclomatic_complexity(code, "python") == 1

    def test_line_numbers_are_preserved(self):
        """Cognitive and nesting are line-oriented, so spans must keep newlines."""
        code = 'x = 1\n"""\na\nb\n"""\ny = 2\n'

        assert len(strip_non_code(code).split("\n")) == len(code.split("\n"))

    def test_docstring_does_not_inflate_nesting(self):
        code = 'def f():\n    """\n            deeply indented prose\n    """\n    return 1\n'

        assert calculate_nesting_depth(code, "python") <= 1

    def test_docstring_does_not_inflate_cognitive(self):
        documented = 'def f(x):\n    """Do it for each one and then, if needed, or else stop."""\n    return x\n'

        assert calculate_cognitive_complexity(documented, "python") == 0

    def test_hash_inside_string_is_not_treated_as_comment(self):
        code = 'def f():\n    return "#" + "value"\n'

        assert strip_non_code(code).count("\n") == code.count("\n")

    def test_js_block_and_line_comments_are_stripped(self):
        code = "function f(x) {\n  /* if (a && b) for */\n  // while (c || d)\n  return x;\n}\n"

        assert calculate_cyclomatic_complexity(code, "typescript") == 1
