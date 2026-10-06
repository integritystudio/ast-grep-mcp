"""Security vulnerability scanner for code quality.

This module provides functionality to detect common security vulnerabilities:
- SQL injection
- Cross-site scripting (XSS)
- Command injection
- Hardcoded secrets
- Insecure cryptography
- Path traversal
- Unsafe deserialization
"""

import copy
import re
import time
from pathlib import Path
from typing import Any, Dict, List, cast

import sentry_sdk

from ast_grep_mcp.constants import (
    ConversionFactors,
    FilePatterns,
    PatternSuggestionConfidence,
    SecurityScanDefaults,
    SeverityRankingDefaults,
)
from ast_grep_mcp.core.executor import stream_ast_grep_results
from ast_grep_mcp.core.logging import get_logger
from ast_grep_mcp.models.standards import SecurityIssue, SecurityScanResult

logger = get_logger(__name__)

# =============================================================================
# Scan Configuration
# =============================================================================

SCAN_CONFIG = {
    "sql_injection": {"patterns_dict": "SQL_INJECTION_PATTERNS", "use_ast_grep": True},
    "xss": {"patterns_dict": "XSS_PATTERNS", "use_ast_grep": True},
    "command_injection": {"patterns_dict": "COMMAND_INJECTION_PATTERNS", "use_ast_grep": True},
    "hardcoded_secrets": {"patterns_dict": None, "use_regex": True},
    "insecure_crypto": {"patterns_dict": "CRYPTO_PATTERNS", "use_ast_grep": True},
}

# =============================================================================
# Vulnerability Patterns - SQL Injection
# =============================================================================

SQL_INJECTION_PATTERNS = {
    "python": [
        {
            "pattern": "cursor.execute(f$$$)",
            "severity": "critical",
            "title": "SQL Injection via f-string",
            "description": "Using f-strings or string formatting in SQL queries allows SQL injection attacks",
            "remediation": "Use parameterized queries with placeholders (?, %s, etc.)",
            "cwe": "CWE-89",
            "confidence": SecurityScanDefaults.HIGH_CONFIDENCE,
        },
        {
            "pattern": "cursor.execute($STR.format($$$))",
            "severity": "critical",
            "title": "SQL Injection via .format()",
            "description": "Using .format() in SQL queries allows SQL injection attacks",
            "remediation": "Use parameterized queries instead of string formatting",
            "cwe": "CWE-89",
            "confidence": SecurityScanDefaults.HIGH_CONFIDENCE,
        },
        {
            "pattern": "cursor.execute($STR + $$$)",
            "severity": "critical",
            "title": "SQL Injection via string concatenation",
            "description": "Concatenating user input into SQL queries allows SQL injection",
            "remediation": "Use parameterized queries with ? or %s placeholders",
            "cwe": "CWE-89",
            "confidence": SecurityScanDefaults.ELEVATED_CONFIDENCE,
        },
    ],
    "javascript": [
        {
            "pattern": "db.query(`$$$${$VAR}$$$`)",
            "severity": "critical",
            "title": "SQL Injection via template literal",
            "description": "Using template literals with variables in SQL queries enables SQL injection",
            "remediation": "Use parameterized queries or prepared statements",
            "cwe": "CWE-89",
            "confidence": SecurityScanDefaults.HIGH_CONFIDENCE,
        }
    ],
    "typescript": [
        {
            "pattern": "db.query(`$$$${$VAR}$$$`)",
            "severity": "critical",
            "title": "SQL Injection via template literal",
            "description": "Using template literals with variables in SQL queries enables SQL injection",
            "remediation": "Use parameterized queries or prepared statements",
            "cwe": "CWE-89",
            "confidence": SecurityScanDefaults.HIGH_CONFIDENCE,
        }
    ],
}

# =============================================================================
# Vulnerability Patterns - XSS (Cross-Site Scripting)
# =============================================================================

XSS_PATTERNS = {
    "python": [
        {
            "pattern": 'f"<$$$>{$VAR}<$$$>"',
            "severity": "high",
            "title": "XSS via unescaped HTML in f-string",
            "description": "Inserting user input directly into HTML without escaping enables XSS attacks",
            "remediation": "Use HTML escaping functions like html.escape() or template engine auto-escaping",
            "cwe": "CWE-79",
            "confidence": SecurityScanDefaults.MEDIUM_CONFIDENCE,
        }
    ],
    "javascript": [
        {
            "pattern": "innerHTML = $VAR",
            "severity": "high",
            "title": "XSS via innerHTML",
            "description": "Assigning user input to innerHTML without sanitization enables XSS",
            "remediation": "Use textContent or sanitize HTML with DOMPurify",
            "cwe": "CWE-79",
            "confidence": SecurityScanDefaults.DEFAULT_CONFIDENCE,
        },
        {
            "pattern": "document.write($VAR)",
            "severity": "high",
            "title": "XSS via document.write()",
            "description": "Using document.write() with user input can enable XSS attacks",
            "remediation": "Avoid document.write() or sanitize input properly",
            "cwe": "CWE-79",
            "confidence": SecurityScanDefaults.ELEVATED_CONFIDENCE,
        },
    ],
    "typescript": [
        {
            "pattern": "innerHTML = $VAR",
            "severity": "high",
            "title": "XSS via innerHTML",
            "description": "Assigning user input to innerHTML without sanitization enables XSS",
            "remediation": "Use textContent or sanitize HTML with DOMPurify",
            "cwe": "CWE-79",
            "confidence": SecurityScanDefaults.DEFAULT_CONFIDENCE,
        }
    ],
}

# =============================================================================
# Vulnerability Patterns - Command Injection
# =============================================================================

COMMAND_INJECTION_PATTERNS = {
    "python": [
        {
            "pattern": "os.system(f$$$)",
            "severity": "critical",
            "title": "Command Injection via os.system() with f-string",
            "description": "Using os.system() with formatted strings allows command injection",
            "remediation": "Use subprocess.run() with list of arguments instead",
            "cwe": "CWE-78",
            "confidence": SecurityScanDefaults.VERY_HIGH_CONFIDENCE,
        },
        {
            "pattern": "os.system($STR + $$$)",
            "severity": "critical",
            "title": "Command Injection via os.system() concatenation",
            "description": "String concatenation in os.system() enables command injection",
            "remediation": "Use subprocess.run(['cmd', arg1, arg2]) with array form",
            "cwe": "CWE-78",
            "confidence": SecurityScanDefaults.HIGH_CONFIDENCE,
        },
        {
            "pattern": "subprocess.run($STR, shell=True)",
            "severity": "high",
            "title": "Command Injection via subprocess with shell=True",
            "description": "Using shell=True with user input allows command injection",
            "remediation": "Use subprocess.run() with list form and shell=False",
            "cwe": "CWE-78",
            "confidence": SecurityScanDefaults.ELEVATED_CONFIDENCE,
        },
        {
            "pattern": "eval($VAR)",
            "severity": "critical",
            "title": "Code Injection via eval()",
            "description": "eval() with user input allows arbitrary code execution",
            "remediation": "Never use eval() with untrusted input; use safe alternatives like ast.literal_eval()",
            "cwe": "CWE-95",
            "confidence": 1.0,
        },
        {
            "pattern": "exec($VAR)",
            "severity": "critical",
            "title": "Code Injection via exec()",
            "description": "exec() with user input allows arbitrary code execution",
            "remediation": "Never use exec() with untrusted input",
            "cwe": "CWE-95",
            "confidence": 1.0,
        },
    ],
    "javascript": [
        {
            "pattern": "eval($VAR)",
            "severity": "critical",
            "title": "Code Injection via eval()",
            "description": "eval() with user input allows arbitrary code execution",
            "remediation": "Never use eval(); use JSON.parse() for JSON or safer alternatives",
            "cwe": "CWE-95",
            "confidence": 1.0,
        }
    ],
}

# =============================================================================
# Vulnerability Patterns - Hardcoded Secrets
# =============================================================================

SECRET_REGEX_PATTERNS = [
    {
        "regex": r'["\']sk-[a-zA-Z0-9]{32,}["\']',
        "severity": "critical",
        "title": "Hardcoded API Key (OpenAI format)",
        "description": "Hardcoded API key found in source code",
        "remediation": "Use environment variables or secret management system",
        "cwe": "CWE-798",
    },
    {
        "regex": r'["\']AIza[a-zA-Z0-9_-]{35}["\']',
        "severity": "critical",
        "title": "Hardcoded Google API Key",
        "description": "Hardcoded Google API key found in source code",
        "remediation": "Use environment variables or Google Secret Manager",
        "cwe": "CWE-798",
    },
    {
        "regex": r'["\']ghp_[a-zA-Z0-9]{36}["\']',
        "severity": "critical",
        "title": "Hardcoded GitHub Personal Access Token",
        "description": "GitHub token found hardcoded in source code",
        "remediation": "Use environment variables or GitHub Secrets",
        "cwe": "CWE-798",
    },
    {
        "regex": r'password\s*=\s*["\'][^"\']{8,}["\']',
        "severity": "high",
        "title": "Hardcoded Password",
        "description": "Password appears to be hardcoded in source code",
        "remediation": "Use environment variables or credential management system",
        "cwe": "CWE-798",
    },
    {
        "regex": r"Bearer\s+[a-zA-Z0-9_-]{20,}",
        "severity": "high",
        "title": "Hardcoded Bearer Token",
        "description": "Bearer token found in source code",
        "remediation": "Use environment variables for authentication tokens",
        "cwe": "CWE-798",
    },
]

# =============================================================================
# Vulnerability Patterns - Insecure Cryptography
# =============================================================================

CRYPTO_PATTERNS = {
    "python": [
        {
            "pattern": "hashlib.md5($$$)",
            "severity": "medium",
            "title": "Insecure Hash Algorithm (MD5)",
            "description": "MD5 is cryptographically broken and should not be used for security",
            "remediation": "Use bcrypt, argon2, or scrypt for password hashing; SHA-256+ for checksums",
            "cwe": "CWE-327",
            "confidence": SecurityScanDefaults.HIGH_CONFIDENCE,
        },
        {
            "pattern": "hashlib.sha1($$$)",
            "severity": "medium",
            "title": "Weak Hash Algorithm (SHA-1)",
            "description": "SHA-1 is weak and deprecated for security purposes",
            "remediation": "Use SHA-256 or stronger; use bcrypt/argon2 for passwords",
            "cwe": "CWE-327",
            "confidence": SecurityScanDefaults.HIGH_CONFIDENCE,
        },
    ]
}

# =============================================================================
# Scanner Implementation
# =============================================================================


def _match_to_issue(match: Dict[str, Any], pattern_def: Dict[str, Any]) -> SecurityIssue:
    range_info = match.get("range", {})
    start = range_info.get("start", {})
    end = range_info.get("end", {})
    return SecurityIssue(
        file=match.get("file", ""),
        line=start.get("line", 0) + 1,
        column=start.get("column", 0) + 1,
        end_line=end.get("line", 0) + 1,
        end_column=end.get("column", 0) + 1,
        issue_type=pattern_def.get("issue_type", "unknown"),
        severity=pattern_def["severity"],
        title=pattern_def["title"],
        description=pattern_def["description"],
        code_snippet=match.get("text", ""),
        remediation=pattern_def["remediation"],
        cwe_id=pattern_def.get("cwe"),
        confidence=pattern_def.get("confidence", SecurityScanDefaults.DEFAULT_CONFIDENCE),
    )


def scan_for_vulnerability(project_folder: str, language: str, patterns: List[Dict[str, Any]]) -> List[SecurityIssue]:
    """Scan for vulnerabilities using ast-grep patterns.

    Args:
        project_folder: Project root directory
        language: Programming language
        patterns: List of vulnerability patterns to check

    Returns:
        List of SecurityIssue objects
    """
    issues = []

    for pattern_def in patterns:
        try:
            args = ["-p", pattern_def["pattern"], "-l", language, "--json=stream", project_folder]
            results = stream_ast_grep_results("run", args, max_results=0)
            for match in results:
                issues.append(_match_to_issue(match, pattern_def))
        except Exception as e:
            logger.warning(f"Pattern scan failed: {pattern_def.get('title')}: {e}")
            sentry_sdk.capture_exception(e)

    return issues


def scan_for_secrets_regex(project_folder: str, language: str) -> List[SecurityIssue]:
    """Scan for hardcoded secrets using regex patterns.

    Args:
        project_folder: Project root directory
        language: Programming language (for file filtering)

    Returns:
        List of SecurityIssue objects
    """
    issues = []
    project_path = Path(project_folder)

    # Get file extensions for language
    exts = _get_language_extensions(language)

    # Scan files by extension
    for ext in exts:
        file_issues = _scan_files_for_secrets(project_path, ext)
        issues.extend(file_issues)

    return issues


def _get_language_extensions(language: str) -> List[str]:
    """Get file extensions for a programming language.

    Args:
        language: Programming language name

    Returns:
        List of file extensions
    """
    extensions = {"python": [".py"], "javascript": [".js"], "typescript": [".ts", ".tsx"], "java": [".java"]}
    return extensions.get(language, [".py", ".js", ".ts", ".java"])


def _should_skip_file(file_path: Path, project_path: Path) -> bool:
    """Check if a file should be skipped during scanning.

    Args:
        file_path: Path to check
        project_path: Project root; only directories below it are considered

    Returns:
        True if file lies under a skipped directory (whole path segments, not substrings)
    """
    return any(part in FilePatterns.SKIP_DIR_NAMES for part in file_path.relative_to(project_path).parts[:-1])


def _scan_files_for_secrets(project_path: Path, ext: str) -> List[SecurityIssue]:
    """Scan all files with given extension for secrets.

    Args:
        project_path: Project root path
        ext: File extension to scan

    Returns:
        List of security issues found
    """
    issues = []

    for file_path in project_path.rglob(f"*{ext}"):
        if _should_skip_file(file_path, project_path):
            continue

        file_issues = _scan_single_file_for_secrets(file_path)
        issues.extend(file_issues)

    return issues


def _scan_single_file_for_secrets(file_path: Path) -> List[SecurityIssue]:
    """Scan a single file for hardcoded secrets.

    Args:
        file_path: Path to file to scan

    Returns:
        List of security issues found in file
    """
    issues = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()

        # Check each pattern against file content
        for pattern_def in SECRET_REGEX_PATTERNS:
            pattern_issues = _scan_lines_for_pattern(lines, pattern_def, str(file_path))
            issues.extend(pattern_issues)

    except Exception as e:
        logger.warning(f"Secret scan failed for {file_path}: {e}")

    return issues


def _scan_lines_for_pattern(lines: List[str], pattern_def: Dict[str, Any], file_path: str) -> List[SecurityIssue]:
    """Scan lines for a specific secret pattern.

    Args:
        lines: File lines to scan
        pattern_def: Pattern definition with regex and metadata
        file_path: Path to file being scanned

    Returns:
        List of security issues found
    """
    issues = []

    for line_num, line in enumerate(lines, start=1):
        matches = re.finditer(pattern_def["regex"], line, re.IGNORECASE)

        for match in matches:
            issue = _create_secret_issue(file_path=file_path, line_num=line_num, line=line, match=match, pattern_def=pattern_def)
            issues.append(issue)

    return issues


def _create_secret_issue(file_path: str, line_num: int, line: str, match: re.Match[str], pattern_def: Dict[str, Any]) -> SecurityIssue:
    """Create a SecurityIssue for a found secret.

    Args:
        file_path: Path to file containing the secret
        line_num: Line number where secret found
        line: The line containing the secret
        match: Regex match object
        pattern_def: Pattern definition with metadata

    Returns:
        SecurityIssue object
    """
    return SecurityIssue(
        file=file_path,
        line=line_num,
        column=match.start() + 1,
        end_line=line_num,
        end_column=match.end() + 1,
        issue_type="hardcoded_secret",
        severity=pattern_def["severity"],
        title=pattern_def["title"],
        description=pattern_def["description"],
        code_snippet=line.strip(),
        remediation=pattern_def["remediation"],
        cwe_id=pattern_def.get("cwe"),
        confidence=PatternSuggestionConfidence.PATTERN_MATCH,
    )


# =============================================================================
# Helper Functions for Complexity Reduction
# =============================================================================


def _scan_for_issue_type(issue_type: str, config: Dict[str, Any], project_folder: str, language: str) -> List[SecurityIssue]:
    """Scan for a specific vulnerability type.

    Args:
        issue_type: Type of vulnerability to scan for
        config: Configuration for this issue type from SCAN_CONFIG
        project_folder: Project root directory
        language: Programming language

    Returns:
        List of security issues found
    """
    # Handle regex-based scanning (e.g., hardcoded secrets)
    if config.get("use_regex"):
        return scan_for_secrets_regex(project_folder, language)

    # Handle ast-grep based scanning
    patterns_dict_name = config.get("patterns_dict")
    if not patterns_dict_name:
        return []

    # Get the patterns dictionary by name
    patterns_dict = globals()[patterns_dict_name]
    if language not in patterns_dict:
        return []

    # Deep copy patterns and add issue type
    patterns = copy.deepcopy(patterns_dict[language])
    for p in patterns:
        p["issue_type"] = issue_type

    return scan_for_vulnerability(project_folder, language, patterns)


def _filter_by_severity(issues: List[SecurityIssue], severity_threshold: str, max_issues: int) -> List[SecurityIssue]:
    """Filter issues by severity threshold and limit count.

    Args:
        issues: All security issues found
        severity_threshold: Minimum severity to include
        max_issues: Maximum number of issues to return (0 = unlimited)

    Returns:
        Filtered list of security issues
    """
    severity_order = SeverityRankingDefaults.SECURITY_SCAN_ORDER
    threshold_level = severity_order.get(severity_threshold, 0)

    filtered = [issue for issue in issues if severity_order.get(issue.severity, 0) >= threshold_level]

    if max_issues > 0:
        filtered = filtered[:max_issues]

    return filtered


def _group_issues(issues: List[SecurityIssue]) -> tuple[Dict[str, List[SecurityIssue]], Dict[str, List[SecurityIssue]]]:
    """Group issues by severity and type.

    Args:
        issues: List of security issues to group

    Returns:
        Tuple of (issues_by_severity, issues_by_type)
    """
    by_severity: Dict[str, List[SecurityIssue]] = {}
    by_type: Dict[str, List[SecurityIssue]] = {}

    for issue in issues:
        # Group by severity
        if issue.severity not in by_severity:
            by_severity[issue.severity] = []
        by_severity[issue.severity].append(issue)

        # Group by type
        if issue.issue_type not in by_type:
            by_type[issue.issue_type] = []
        by_type[issue.issue_type].append(issue)

    return by_severity, by_type


def _build_summary(
    by_severity: Dict[str, List[SecurityIssue]], by_type: Dict[str, List[SecurityIssue]], total_count: int
) -> Dict[str, Any]:
    """Build summary statistics for the scan results.

    Args:
        by_severity: Issues grouped by severity
        by_type: Issues grouped by type
        total_count: Total number of issues

    Returns:
        Summary dictionary with counts and statistics
    """
    return {
        "total_issues": total_count,
        "critical_count": len(by_severity.get("critical", [])),
        "high_count": len(by_severity.get("high", [])),
        "medium_count": len(by_severity.get("medium", [])),
        "low_count": len(by_severity.get("low", [])),
        "issue_types_found": list(by_type.keys()),
    }


def _collect_issues(project_folder: str, language: str, issue_types: List[str]) -> List[SecurityIssue]:
    scan_all = "all" in issue_types
    types_to_scan = list(SCAN_CONFIG.keys()) if scan_all else issue_types
    all_issues: List[SecurityIssue] = []
    for issue_type in types_to_scan:
        if issue_type in SCAN_CONFIG:
            issues = _scan_for_issue_type(
                issue_type=issue_type,
                config=cast(Dict[str, Any], SCAN_CONFIG[issue_type]),
                project_folder=project_folder,
                language=language,
            )
            all_issues.extend(issues)
    return all_issues


def detect_security_issues_impl(
    project_folder: str,
    language: str,
    issue_types: List[str] = ["all"],
    severity_threshold: str = "low",
    max_issues: int = SecurityScanDefaults.MAX_ISSUES,
) -> SecurityScanResult:
    """Scan project for security vulnerabilities.

    Args:
        project_folder: Project root directory
        language: Programming language
        issue_types: Types to scan for or ["all"]
        severity_threshold: Minimum severity to report
        max_issues: Maximum issues to find (0 = unlimited)

    Returns:
        SecurityScanResult with all findings
    """
    start_time = time.time()
    all_issues = _collect_issues(project_folder, language, issue_types)
    filtered_issues = _filter_by_severity(issues=all_issues, severity_threshold=severity_threshold, max_issues=max_issues)
    by_severity, by_type = _group_issues(filtered_issues)
    summary = _build_summary(by_severity=by_severity, by_type=by_type, total_count=len(filtered_issues))
    execution_time = int((time.time() - start_time) * ConversionFactors.MILLISECONDS_PER_SECOND)
    return SecurityScanResult(
        summary=summary,
        issues=filtered_issues,
        issues_by_severity=by_severity,
        issues_by_type=by_type,
        files_scanned=len(set(issue.file for issue in filtered_issues)),
        execution_time_ms=execution_time,
    )
