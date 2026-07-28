"""Tests for usage tracking and cost monitoring.

Tests the SQLite-based usage tracking system including:
- Cost calculation
- Usage logging
- Statistics aggregation
- Alert generation
- Tracking decorators and context managers
"""

import os
import tempfile
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from ast_grep_mcp.core.usage_tracking import (
    AlertThresholds,
    OperationType,
    UsageDatabase,
    UsageLogEntry,
    UsageStats,
    calculate_operation_cost,
    format_usage_report,
    track_operation,
    track_usage,
)


class TestOperationPricing:
    """Tests for operation cost calculation."""

    def test_base_cost_only(self):
        """Operations with no additional metrics should use base cost."""
        cost = calculate_operation_cost(OperationType.VALIDATE_SCHEMA)
        assert cost == 0.005  # Base cost for VALIDATE_SCHEMA

    def test_cost_with_files(self):
        """Cost should increase with files processed."""
        base_cost = calculate_operation_cost(OperationType.SEARCH_CODE)
        with_files = calculate_operation_cost(
            OperationType.SEARCH_CODE,
            files_processed=100,
        )
        assert with_files > base_cost
        # 0.001 base + 100 * 0.0001 = 0.011
        assert abs(with_files - 0.011) < 0.0001

    def test_cost_with_lines(self):
        """Cost should increase with lines analyzed."""
        base_cost = calculate_operation_cost(OperationType.FIND_DUPLICATION)
        with_lines = calculate_operation_cost(
            OperationType.FIND_DUPLICATION,
            lines_analyzed=10000,
        )
        assert with_lines > base_cost

    def test_cost_with_matches(self):
        """Cost should increase with matches found."""
        base_cost = calculate_operation_cost(OperationType.SEARCH_CODE)
        with_matches = calculate_operation_cost(
            OperationType.SEARCH_CODE,
            matches_found=50,
        )
        assert with_matches > base_cost

    def test_unknown_operation_uses_default(self):
        """Unknown operations should use default pricing."""
        cost = calculate_operation_cost(OperationType.UNKNOWN)
        assert cost == 0.01  # Default base cost

    def test_refactoring_operations_cost_more(self):
        """Refactoring operations should have higher base costs."""
        search_cost = calculate_operation_cost(OperationType.SEARCH_CODE)
        refactor_cost = calculate_operation_cost(OperationType.EXTRACT_FUNCTION)
        assert refactor_cost > search_cost


class TestUsageLogEntry:
    """Tests for UsageLogEntry model."""

    def test_default_values(self):
        """Entry should have sensible defaults."""
        entry = UsageLogEntry(tool_name="test_tool")
        assert entry.tool_name == "test_tool"
        assert entry.operation_type == OperationType.UNKNOWN
        assert entry.success is True
        assert entry.error_message is None
        assert entry.response_time_ms == 0
        assert entry.estimated_cost == 0.0
        assert entry.id is not None  # Auto-generated

    def test_custom_values(self):
        """Entry should accept custom values."""
        entry = UsageLogEntry(
            tool_name="find_duplication",
            operation_type=OperationType.FIND_DUPLICATION,
            success=False,
            error_message="Test error",
            response_time_ms=1500,
            estimated_cost=0.025,
            files_processed=10,
            matches_found=5,
        )
        assert entry.operation_type == OperationType.FIND_DUPLICATION
        assert entry.success is False
        assert entry.error_message == "Test error"
        assert entry.response_time_ms == 1500


class TestUsageDatabase:
    """Tests for SQLite usage database."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        db = UsageDatabase(db_path)
        yield db
        os.unlink(db_path)

    def test_database_creation(self, temp_db):
        """Database should be created with proper schema."""
        assert os.path.exists(temp_db.db_path)

    def test_log_usage(self, temp_db):
        """Should successfully log usage entries."""
        entry = UsageLogEntry(
            tool_name="test_tool",
            operation_type=OperationType.SEARCH_CODE,
            response_time_ms=100,
            estimated_cost=0.01,
        )
        temp_db.log_usage(entry)

        # Verify entry was logged
        logs = temp_db.get_recent_logs(limit=1)
        assert len(logs) == 1
        assert logs[0].tool_name == "test_tool"
        assert logs[0].operation_type == OperationType.SEARCH_CODE

    def test_log_failure(self, temp_db):
        """Should log failed operations."""
        entry = UsageLogEntry(
            tool_name="failing_tool",
            success=False,
            error_message="Something went wrong",
        )
        temp_db.log_usage(entry)

        logs = temp_db.get_recent_logs(success=False)
        assert len(logs) == 1
        assert logs[0].success is False
        assert "Something went wrong" in logs[0].error_message

    def test_get_stats_empty(self, temp_db):
        """Stats should work with empty database."""
        stats = temp_db.get_stats()
        assert stats.total_calls == 0
        assert stats.success_rate == 0.0
        assert stats.total_cost == 0.0

    def test_get_stats_with_data(self, temp_db):
        """Stats should aggregate data correctly."""
        # Log some entries
        for i in range(10):
            entry = UsageLogEntry(
                tool_name="tool_a" if i < 7 else "tool_b",
                operation_type=OperationType.SEARCH_CODE,
                success=i < 8,  # 2 failures
                estimated_cost=0.01,
                response_time_ms=100,
            )
            temp_db.log_usage(entry)

        stats = temp_db.get_stats()
        assert stats.total_calls == 10
        assert stats.successful_calls == 8
        assert stats.failed_calls == 2
        assert stats.success_rate == 80.0
        assert abs(stats.total_cost - 0.10) < 0.001
        assert "tool_a" in stats.calls_by_tool
        assert stats.calls_by_tool["tool_a"] == 7

    def test_get_stats_time_filter(self, temp_db):
        """Stats should filter by time range."""
        # Log entry now
        entry = UsageLogEntry(tool_name="recent_tool")
        temp_db.log_usage(entry)

        # Query for last hour only
        stats = temp_db.get_stats(
            start_time=datetime.now(UTC) - timedelta(hours=1),
        )
        assert stats.total_calls == 1

        # Query for future (should have no results)
        future_stats = temp_db.get_stats(
            start_time=datetime.now(UTC) + timedelta(hours=1),
        )
        assert future_stats.total_calls == 0

    def test_filter_by_tool(self, temp_db):
        """Should filter logs by tool name."""
        temp_db.log_usage(UsageLogEntry(tool_name="tool_a"))
        temp_db.log_usage(UsageLogEntry(tool_name="tool_b"))
        temp_db.log_usage(UsageLogEntry(tool_name="tool_a"))

        logs_a = temp_db.get_recent_logs(tool_name="tool_a")
        assert len(logs_a) == 2

        logs_b = temp_db.get_recent_logs(tool_name="tool_b")
        assert len(logs_b) == 1


class TestUsageAlerts:
    """Tests for usage alert generation."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        db = UsageDatabase(db_path)
        yield db
        os.unlink(db_path)

    def test_no_alerts_when_under_threshold(self, temp_db):
        """No alerts when usage is under all thresholds."""
        # Log a few normal calls
        for _ in range(5):
            temp_db.log_usage(
                UsageLogEntry(
                    tool_name="test",
                    estimated_cost=0.001,
                )
            )

        alerts = temp_db.get_alerts()
        assert len(alerts) == 0

    def test_daily_calls_warning(self, temp_db):
        """Should generate warning when daily calls exceed warning threshold."""
        thresholds = AlertThresholds(daily_calls_warning=10, daily_calls_critical=100)

        # Log enough calls to trigger warning
        for _ in range(15):
            temp_db.log_usage(UsageLogEntry(tool_name="test"))

        alerts = temp_db.get_alerts(thresholds)
        assert any(a.level == "warning" and a.metric == "daily_calls" for a in alerts)

    def test_daily_calls_critical(self, temp_db):
        """Should generate critical alert when daily calls exceed critical threshold."""
        thresholds = AlertThresholds(daily_calls_warning=5, daily_calls_critical=10)

        for _ in range(15):
            temp_db.log_usage(UsageLogEntry(tool_name="test"))

        alerts = temp_db.get_alerts(thresholds)
        assert any(a.level == "critical" and a.metric == "daily_calls" for a in alerts)

    def test_failure_rate_alert(self, temp_db):
        """Should generate alert when failure rate is high."""
        thresholds = AlertThresholds(
            failure_rate_warning=0.1,
            failure_rate_critical=0.3,
        )

        # Log 10 calls, 4 failures = 40% failure rate
        for i in range(10):
            temp_db.log_usage(
                UsageLogEntry(
                    tool_name="test",
                    success=i >= 4,  # First 4 fail
                )
            )

        alerts = temp_db.get_alerts(thresholds)
        assert any(a.metric == "failure_rate" for a in alerts)

    def test_hourly_failures_alert(self, temp_db):
        """Should alert on high hourly failure count."""
        thresholds = AlertThresholds(
            hourly_failures_warning=5,
            hourly_failures_critical=10,
        )

        # Log 7 failures
        for _ in range(7):
            temp_db.log_usage(
                UsageLogEntry(
                    tool_name="test",
                    success=False,
                    error_message="Error",
                )
            )

        alerts = temp_db.get_alerts(thresholds)
        assert any(a.metric == "hourly_failures" for a in alerts)


class TestTrackOperation:
    """Tests for track_operation context manager."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database and patch global."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        db = UsageDatabase(db_path)

        with patch("ast_grep_mcp.core.usage_tracking.get_usage_database", return_value=db):
            yield db

        os.unlink(db_path)

    def test_successful_operation(self, temp_db):
        """Should log successful operations."""
        with track_operation("test_tool", OperationType.SEARCH_CODE) as tracker:
            tracker.files_processed = 10
            tracker.matches_found = 5

        logs = temp_db.get_recent_logs(limit=1)
        assert len(logs) == 1
        assert logs[0].tool_name == "test_tool"
        assert logs[0].success is True
        assert logs[0].files_processed == 10
        assert logs[0].matches_found == 5

    def test_failed_operation(self, temp_db):
        """Should log failed operations with error message."""
        try:
            with track_operation("failing_tool", OperationType.APPLY_REWRITE):
                raise ValueError("Test error")
        except ValueError:
            pass

        logs = temp_db.get_recent_logs(limit=1)
        assert len(logs) == 1
        assert logs[0].success is False
        assert "Test error" in logs[0].error_message

    def test_response_time_tracking(self, temp_db):
        """Should track response time."""
        with track_operation("slow_tool", OperationType.ANALYZE_COMPLEXITY):
            time.sleep(0.1)  # 100ms

        logs = temp_db.get_recent_logs(limit=1)
        assert logs[0].response_time_ms >= 100

    def test_cost_calculation(self, temp_db):
        """Should calculate cost based on metrics."""
        with track_operation("cost_tool", OperationType.FIND_DUPLICATION) as tracker:
            tracker.files_processed = 50
            tracker.lines_analyzed = 1000

        logs = temp_db.get_recent_logs(limit=1)
        # Base 0.01 + 50*0.001 + 1000*0.00001 = 0.01 + 0.05 + 0.01 = 0.07
        assert logs[0].estimated_cost > 0.01


class TestTrackUsageDecorator:
    """Tests for @track_usage decorator."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database and patch global."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        db = UsageDatabase(db_path)

        with patch("ast_grep_mcp.core.usage_tracking.get_usage_database", return_value=db):
            yield db

        os.unlink(db_path)

    def test_decorator_logs_success(self, temp_db):
        """Decorator should log successful function calls."""

        @track_usage("decorated_func", OperationType.SEARCH_CODE)
        def sample_function():
            return {"matches": [1, 2, 3], "files_processed": 5}

        result = sample_function()
        assert result["matches"] == [1, 2, 3]

        logs = temp_db.get_recent_logs(limit=1)
        assert len(logs) == 1
        assert logs[0].tool_name == "decorated_func"
        assert logs[0].success is True

    def test_decorator_logs_failure(self, temp_db):
        """Decorator should log failed function calls."""

        @track_usage("failing_func", OperationType.APPLY_REWRITE)
        def failing_function():
            raise RuntimeError("Intentional failure")

        with pytest.raises(RuntimeError):
            failing_function()

        logs = temp_db.get_recent_logs(limit=1)
        assert len(logs) == 1
        assert logs[0].success is False
        assert "Intentional failure" in logs[0].error_message

    def test_decorator_extracts_metrics(self, temp_db):
        """Decorator should extract metrics from return value."""

        @track_usage("metrics_func", OperationType.FIND_DUPLICATION)
        def function_with_metrics():
            return {
                "files_processed": 10,
                "lines_analyzed": 500,
                "matches": [{"id": 1}, {"id": 2}],
            }

        function_with_metrics()

        logs = temp_db.get_recent_logs(limit=1)
        assert logs[0].files_processed == 10
        assert logs[0].matches_found == 2


class TestFormatUsageReport:
    """Tests for usage report formatting."""

    def test_basic_report(self):
        """Should format a basic report."""
        stats = UsageStats(
            period_start=datetime(2025, 1, 1),
            period_end=datetime(2025, 1, 7),
            total_calls=100,
            successful_calls=95,
            failed_calls=5,
            success_rate=95.0,
            total_cost=1.5,
            average_cost=0.015,
            average_response_time_ms=150.0,
        )

        report = format_usage_report(stats)

        assert "USAGE STATISTICS REPORT" in report
        assert "Total Calls:      100" in report
        assert "Success Rate:     95.0%" in report
        assert "Total Cost:       1.500000 units" in report

    def test_report_with_tools(self):
        """Should include tool breakdown in report."""
        stats = UsageStats(
            period_start=datetime(2025, 1, 1),
            period_end=datetime(2025, 1, 7),
            calls_by_tool={"find_duplication": 50, "search_code": 30},
            cost_by_tool={"find_duplication": 0.5, "search_code": 0.3},
        )

        report = format_usage_report(stats)

        assert "CALLS BY TOOL" in report
        assert "find_duplication" in report
        assert "search_code" in report


class TestIntegrationWithDetector:
    """Integration tests with DuplicationDetector."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database and patch global."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        db = UsageDatabase(db_path)

        with patch("ast_grep_mcp.core.usage_tracking.get_usage_database", return_value=db):
            yield db

        os.unlink(db_path)

    def test_detector_logs_usage(self, temp_db):
        """DuplicationDetector should log usage when called."""
        from ast_grep_mcp.features.deduplication.detector import DuplicationDetector

        # Create a mock project folder that doesn't require real files
        with patch.object(DuplicationDetector, "_find_constructs", return_value=([], 0)):
            detector = DuplicationDetector()
            detector.find_duplication("/fake/path")

        # Verify usage was logged
        logs = temp_db.get_recent_logs(tool_name="find_duplication")
        assert len(logs) == 1
        assert logs[0].operation_type == OperationType.FIND_DUPLICATION
