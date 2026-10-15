"""Offline tests only: no MT5 module, account, process or socket is accessed."""
from copy import deepcopy
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_mt5_quotes.py"
SPEC = importlib.util.spec_from_file_location("offline_mt5_exporter", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
START_MS = 1761955200000
SECRET = "SENTINEL_PRIVATE_LOGIN_NAME_PASSWORD"


def tick(delta=0, **changes):
    value = {"time": (START_MS + delta) // 1000, "time_msc": START_MS + delta,
             "bid": 100.0, "ask": 100.2, "last": 0.0, "volume": 0,
             "flags": 6, "volume_real": 0.0}
    return {**value, **changes}


class FakeBackend:
    COPY_TICKS_ALL = -1

    def __init__(self):
        self.calls = []
        self.account = SimpleNamespace(trade_mode=0, server="Deriv-Demo", company="Deriv (SVG) LLC")
        self.terminal = SimpleNamespace(connected=True, trade_allowed=False, tradeapi_disabled=True)
        self.rows = [tick(), tick(0), tick(500), tick(7000)]
        self.code = 1
        self.after_copy = None
        self.before_shutdown = None

    def __getattr__(self, name):
        if name in {"login", "order_send", "order_check", "symbol_select", "history_deals_get", "positions_get"}:
            raise AssertionError("Forbidden backend attribute accessed: " + name)
        raise AttributeError(name)

    def initialize(self, *args, **kwargs):
        self.calls.append(("initialize", args, kwargs))
        return True

    def account_info(self):
        self.calls.append(("account_info", (), {}))
        return self.account

    def terminal_info(self):
        self.calls.append(("terminal_info", (), {}))
        return self.terminal

    def symbol_info(self, source):
        self.calls.append(("symbol_info", (source,), {}))
        return SimpleNamespace(name=source, custom=False, point=.001, digits=3,
                               trade_contract_size=1.0, trade_tick_size=.001,
                               volume_min=.2, volume_max=100.0, volume_step=.1,
                               trade_tick_value=.001, swap_long=-9.0, swap_short=-23.0,
                               currency_profit="USD", currency_margin="USD")

    def copy_ticks_range(self, *args):
        self.calls.append(("copy_ticks_range", args, {}))
        if self.after_copy:
            self.after_copy()
        return deepcopy(self.rows)

    def last_error(self):
        self.calls.append(("last_error", (), {}))
        return self.code, SECRET

    def shutdown(self):
        self.calls.append(("shutdown", (), {}))
        if self.before_shutdown:
            self.before_shutdown()


@pytest.fixture
def request_data(tmp_path):
    executable = tmp_path / "terminal64.exe"
    executable.write_bytes(b"FAKE-NOT-EXECUTED")
    return {"terminal_path": str(executable), "output": str(tmp_path / "output"),
            "symbols": {"BOOM500": "Boom 500 Index", "CRASH500": "Crash 500 Index"},
            "windows": [["2025-11-01T00:00:00Z", "2025-11-01T00:01:00Z"]],
            "allow_existing_live_market_data": False, "timeout_ms": 1000, "max_call_seconds": 60}


def run(request_data, backend=None, **kwargs):
    backend = FakeBackend() if backend is None else backend
    manifest, code = exporter.export_quotes(
        request_data, backend, platform_name="win32", process_checker=lambda _: True,
        environment={}, now=datetime(2026, 1, 1, tzinfo=timezone.utc), **kwargs)
    return backend, manifest, code


def assert_no_output(request_data):
    assert not Path(request_data["output"]).exists()
    assert not Path(request_data["output"] + ".lock").exists()


def test_success_preserves_duplicates_scope_and_call_allowlist(request_data):
    backend, manifest, code = run(request_data)
    assert code == 0
    assert set(name for name, *_ in backend.calls) <= exporter.ALLOWED_CALLS
    assert backend.calls[0][0] == "initialize"
    assert backend.calls[0][2] == {"timeout": 1000}
    assert len(backend.calls[0][1]) == 1
    assert backend.calls[-1][0] == "shutdown"
    assert set(manifest["safety"]) == set(exporter.SAFETY)
    assert all(v is False for v in manifest["safety"].values())
    assert manifest["coverage_complete_verified"] is False
    assert manifest["historical_costs_verified"] is False
    assert manifest["execution_verified"] is False
    assert manifest["orders_sent"] == 0
    assert manifest["provenance"]["account_mode"] == "DEMO"
    for result in manifest["results"]:
        assert result["received_count"] == 4
        assert result["same_millisecond_adjacent_pairs"] == 1
        assert result["gap_diagnostics"]["intervals_above_threshold"] == 1
        assert result["current_symbol_specification"]["historical_applicability_verified"] is False
        assert result["tick_flag_counts"] == {"6": 4}
        raw = Path(request_data["output"]) / result["files"][0]["name"]
        with raw.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4 and rows[0] == rows[1]
        assert [int(r["time_msc"]) for r in rows] == [START_MS, START_MS, START_MS+500, START_MS+7000]
    for name, args, _ in backend.calls:
        if name == "copy_ticks_range":
            assert args[1].utcoffset().total_seconds() == 0
            assert args[2].isoformat() == "2025-11-01T00:00:59.999000+00:00"
            assert args[3] == -1
    out = Path(request_data["output"])
    assert (out / "manifest.sha256").read_text().strip() == exporter.digest(out / "manifest.json")
    assert all(SECRET not in p.read_text() for p in out.iterdir())
    assert not Path(str(out) + ".lock").exists()


@pytest.mark.parametrize("value", [True, 0, None, "false"])
def test_safety_dict_strict_false_before_connection(request_data, value):
    backend = FakeBackend()
    flags = exporter.SAFETY.copy(); flags["LIVE_ALLOWED"] = value
    with pytest.raises(exporter.ExportError, match="eight safety"):
        run(request_data, backend, safety=flags)
    assert not backend.calls
    assert_no_output(request_data)


@pytest.mark.parametrize("key", list(exporter.SAFETY))
def test_true_environment_flag_before_connection(request_data, key):
    backend = FakeBackend()
    with pytest.raises(exporter.ExportError, match="environment"):
        exporter.export_quotes(request_data, backend, environment={key: "true"}, platform_name="win32")
    assert not backend.calls


@pytest.mark.parametrize("mode", [2, 1, False, None, "0", -1])
def test_real_unknown_and_bool_account_rejected(request_data, mode):
    backend = FakeBackend(); backend.account.trade_mode = mode
    with pytest.raises(exporter.ExportError, match="DEMO"):
        run(request_data, backend)
    assert not any(c[0] == "copy_ticks_range" for c in backend.calls)
    assert backend.calls[-1][0] == "shutdown"
    assert_no_output(request_data)


def test_live_data_requires_explicit_flag_and_remains_read_only(request_data):
    request_data["allow_existing_live_market_data"] = True
    backend = FakeBackend(); backend.account.trade_mode = 2; backend.account.server = "DerivSVG-Server-02"
    _, result, code = run(request_data, backend)
    assert code == 0
    assert result["provenance"]["account_mode"] == "EXISTING_LIVE_MARKET_DATA_ONLY"
    assert result["existing_live_market_data_explicitly_allowed"] is True
    assert result["orders_sent"] == 0 and all(v is False for v in result["safety"].values())


@pytest.mark.parametrize("server", ["NotDeriv-Demo", "OtherBroker", "Derivatives-Demo", SECRET, "Deriv\nDemo"])
def test_server_identity_rejected(request_data, server):
    backend = FakeBackend(); backend.account.server = server
    with pytest.raises(exporter.ExportError, match="server"):
        run(request_data, backend)
    assert_no_output(request_data)


@pytest.mark.parametrize("attribute,value", [("connected", False), ("connected", 1), ("trade_allowed", True),
                                            ("trade_allowed", 0), ("tradeapi_disabled", False), ("tradeapi_disabled", 1),
                                            ("tradeapi_disabled", None)])
def test_terminal_protection_fail_closed(request_data, attribute, value):
    backend = FakeBackend(); setattr(backend.terminal, attribute, value)
    with pytest.raises(exporter.ExportError):
        run(request_data, backend)
    assert_no_output(request_data)


def test_missing_trading_protection_rejected(request_data):
    backend = FakeBackend(); del backend.terminal.trade_allowed
    with pytest.raises(exporter.ExportError): run(request_data, backend)
    assert_no_output(request_data)


def test_unavailable_api_disabled_disclosed_not_fabricated(request_data):
    backend = FakeBackend(); del backend.terminal.tradeapi_disabled
    _, manifest, _ = run(request_data, backend)
    assert manifest["provenance"]["external_python_trading_disabled"] is None
    assert manifest["provenance"]["external_python_trading_state_available"] is False


def test_account_server_change_during_copy_rejected(request_data):
    backend = FakeBackend()
    backend.after_copy = lambda: setattr(backend.account, "server", "Deriv-Server-99")
    with pytest.raises(exporter.ExportError, match="changed"):
        run(request_data, backend)
    assert_no_output(request_data)


@pytest.mark.parametrize('custom', [None, True, 0])
def test_symbol_custom_must_be_explicitly_false(request_data, custom):
    backend = FakeBackend()
    original = backend.symbol_info

    def specification(source):
        value = original(source)
        if custom is None:
            del value.custom
        else:
            value.custom = custom
        return value

    backend.symbol_info = specification
    with pytest.raises(exporter.ExportError, match='custom'):
        run(request_data, backend)
    assert not any(call[0] == 'copy_ticks_range' for call in backend.calls)
    assert_no_output(request_data)


def test_changed_connection_during_current_spec_fails_before_copy(request_data):
    backend = FakeBackend()
    original = backend.symbol_info

    def specification(source):
        value = original(source)
        backend.account.server = 'Deriv-Server-99'
        return value

    backend.symbol_info = specification
    with pytest.raises(exporter.ExportError, match='changed'):
        run(request_data, backend)
    assert not any(call[0] == 'copy_ticks_range' for call in backend.calls)
    assert_no_output(request_data)


def test_trading_enabled_during_copy_rejected(request_data):
    backend = FakeBackend()
    backend.after_copy = lambda: setattr(backend.terminal, "trade_allowed", True)
    with pytest.raises(exporter.ExportError): run(request_data, backend)
    assert_no_output(request_data)


@pytest.mark.parametrize("change", [{"bid": 0}, {"ask": 0}, {"bid": -1}, {"ask": 99}, {"bid": float("nan")},
                                   {"ask": float("inf")}, {"volume": -1}, {"volume": True}, {"flags": -1},
                                   {"time_msc": START_MS+60000}, {"time_msc": START_MS-1}, {"time": START_MS//1000+1}])
def test_bad_prices_times_or_types_rejected(request_data, change):
    backend = FakeBackend(); backend.rows = [tick(**change)]
    _, manifest, code = run(request_data, backend)
    assert code == 2 and manifest["export_status"] == "FAILED_PRESERVED"
    for result in manifest["results"]:
        assert result["status"] == "FAILED_PRESERVED" and result["received_count"] == 1
        assert result["validation_error"] and result["bid_ask_rows_published"] is False
        raw = Path(request_data["output"]) / result["files"][0]["name"]
        with raw.open(newline="") as stream: rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        for key, value in change.items(): assert rows[0][key] == str(value)
        quote = Path(request_data["output"]) / result["files"][1]["name"]
        with quote.open(newline="") as stream: assert list(csv.DictReader(stream)) == []
        assert result["gap_diagnostics"]["available"] is False


def test_order_rows_rejected(request_data):
    backend = FakeBackend(); backend.rows = [{**tick(), "order": 456, "login": SECRET}]
    _, manifest, code = run(request_data, backend)
    assert code == 2 and manifest["export_status"] == "FAILED_PRESERVED"
    for result in manifest["results"]:
        assert "schema" in result["validation_error"]
        assert result["raw_preservation"]["unexpected_or_missing_schema_rows"] == 1
        raw = Path(request_data["output"]) / result["files"][0]["name"]
        with raw.open(newline="") as stream:
            reader = csv.DictReader(stream); rows = list(reader)
            assert set(reader.fieldnames) == set(exporter.RAW_FIELDS)
        assert rows[0]["bid"] == "100.0" and rows[0]["ask"] == "100.2"
    assert all(SECRET not in p.read_text() for p in Path(request_data["output"]).iterdir())


def test_bad_nonnumeric_value_is_redacted_without_price_invention(request_data):
    backend = FakeBackend(); backend.rows = [tick(bid=SECRET), tick(1, ask=SECRET)]
    _, manifest, code = run(request_data, backend)
    assert code == 2
    for result in manifest["results"]:
        assert result["raw_preservation"]["redacted_or_missing_numeric_fields"] == 2
        raw = Path(request_data["output"]) / result["files"][0]["name"]
        with raw.open(newline="") as stream: rows = list(csv.DictReader(stream))
        assert len(rows) == 2
        assert rows[0]["bid"] == "" and rows[0]["ask"] == "100.2"
        assert rows[1]["bid"] == "100.0" and rows[1]["ask"] == ""
    assert all(SECRET not in p.read_text() for p in Path(request_data["output"]).iterdir())


def test_missing_fields_remain_empty_and_failed(request_data):
    backend = FakeBackend(); backend.rows = [tick()]; del backend.rows[0]["bid"]
    _, manifest, code = run(request_data, backend)
    assert code == 2
    assert all(r["status"] == "FAILED_PRESERVED" and r["raw_preservation"]["redacted_or_missing_numeric_fields"] == 1 for r in manifest["results"])


def test_raw_integer_numeric_values_preserved_without_float_coercion(request_data):
    backend = FakeBackend(); backend.rows = [tick(bid=100, ask=101)]
    _, manifest, code = run(request_data, backend)
    assert code == 0
    raw = Path(request_data["output"]) / manifest["results"][0]["files"][0]["name"]
    with raw.open(newline="") as stream: rows = list(csv.DictReader(stream))
    assert rows[0]["bid"] == "100" and rows[0]["ask"] == "101"


def test_unsorted_ticks_rejected_not_sorted(request_data):
    backend = FakeBackend(); backend.rows = [tick(1000), tick(500)]
    _, manifest, code = run(request_data, backend)
    assert code == 2 and manifest["export_status"] == "FAILED_PRESERVED"
    for result in manifest["results"]:
        raw = Path(request_data["output"]) / result["files"][0]["name"]
        with raw.open(newline="") as stream: rows = list(csv.DictReader(stream))
        assert [int(r["time_msc"]) for r in rows] == [START_MS+1000, START_MS+500]


@pytest.mark.parametrize("code,status", [(-10005, "PARTIAL_TIMEOUT_NOT_VERIFIED"), (4403, "PARTIAL_TIMEOUT_NOT_VERIFIED"),
                                        (-10004, "PARTIAL_BACKEND_ERROR_NOT_VERIFIED"), (0, "PARTIAL_BACKEND_ERROR_NOT_VERIFIED")])
def test_partial_error_is_diagnostic_not_success(request_data, code, status):
    backend = FakeBackend(); backend.code = code
    _, manifest, exit_code = run(request_data, backend)
    assert exit_code == 2
    assert manifest["export_status"] == "PARTIAL_OR_EMPTY_NOT_VERIFIED"
    assert all(r["status"] == status and r["received_count"] == 4 for r in manifest["results"])
    assert SECRET not in json.dumps(manifest)


@pytest.mark.parametrize("rows,status", [(None, "ERROR_NO_DATA"), ([], "EMPTY_NOT_VERIFIED")])
def test_no_data_never_claims_success(request_data, rows, status):
    backend = FakeBackend(); backend.rows = rows
    _, manifest, exit_code = run(request_data, backend)
    assert exit_code == 2 and all(r["status"] == status for r in manifest["results"])


def test_late_native_return_preserved_but_partial(request_data):
    moments = iter([0, 61, 70, 131])
    _, manifest, code = run(request_data, monotonic=lambda: next(moments))
    assert code == 2
    assert all(r["status"] == "PARTIAL_TIMEOUT_NOT_VERIFIED" for r in manifest["results"])
    assert "does not interrupt" in manifest["native_call_timeout_limit"]


@pytest.mark.parametrize("windows", [[], [["2025-11-01", "2025-11-02"]],
    [["2025-11-01T00:00:00+03:00", "2025-11-01T00:01:00+03:00"]],
    [["2025-11-01T00:00:00Z", "2025-11-01T00:00:00Z"]],
    [["2025-11-01T00:00:00Z", "2025-11-03T00:00:00Z"]],
    [["2026-02-01T00:00:00Z", "2026-02-01T00:01:00Z"]],
    [["2025-11-01T00:00:00Z", "2025-11-01T00:01:00Z"]]*3,
    [["2025-11-01T00:00:00Z", "2025-11-01T00:01:00Z"], ["2025-11-01T00:00:30Z", "2025-11-01T00:02:00Z"]]])
def test_invalid_windows_before_connection(request_data, windows):
    backend = FakeBackend(); request_data["windows"] = windows
    with pytest.raises(exporter.ExportError): run(request_data, backend)
    assert not backend.calls
    assert_no_output(request_data)


def test_second_window_supported_with_separate_provenance(request_data):
    request_data["windows"].append(["2025-11-02T00:00:00Z", "2025-11-02T00:01:00Z"])
    backend = FakeBackend()
    def copies(source, start, end, flags):
        backend.calls.append(("copy_ticks_range", (source, start, end, flags), {}))
        ms = int(start.timestamp()*1000)
        return [tick(time=ms//1000, time_msc=ms), tick(time=ms//1000, time_msc=ms)]
    backend.copy_ticks_range = copies
    _, manifest, code = run(request_data, backend)
    assert code == 0 and len(manifest["results"]) == 4
    assert sum(r["received_count"] for r in manifest["results"]) == 8


def test_millisecond_boundaries_use_integer_math(request_data):
    request_data["windows"] = [["2025-11-01T00:00:00.001Z", "2025-11-01T00:00:00.003Z"]]
    backend = FakeBackend(); backend.rows = [tick(1), tick(2)]
    _, manifest, code = run(request_data, backend)
    assert code == 0
    for result in manifest["results"]:
        assert result["received_first_time_msc"] == START_MS+1
        assert result["received_last_time_msc"] == START_MS+2
        assert result["gap_diagnostics"]["leading_unobserved_ms"] == 0
        assert result["gap_diagnostics"]["trailing_unobserved_ms"] == 0
        quote = Path(request_data["output"]) / result["files"][1]["name"]
        with quote.open(newline="") as stream: rows = list(csv.DictReader(stream))
        assert rows[0]["utc"] == "2025-11-01T00:00:00.001Z"
        assert rows[1]["utc"] == "2025-11-01T00:00:00.002Z"


def test_existing_output_or_lock_before_connection(request_data):
    backend = FakeBackend()
    output = Path(request_data["output"]); output.mkdir()
    with pytest.raises(exporter.ExportError): run(request_data, backend)
    assert not backend.calls
    output.rmdir()
    lock = Path(str(output)+".lock"); lock.write_text("OTHER_OWNER")
    with pytest.raises(exporter.ExportError, match="lock"):
        run(request_data, backend)
    assert not backend.calls and lock.read_text() == "OTHER_OWNER"


def test_source_tamper_blocks_publication(request_data, monkeypatch):
    original = exporter.digest
    count = 0
    def tampered(path):
        nonlocal count
        if path == SCRIPT.resolve():
            count += 1
            if count > 1: return "changed"
        return original(path)
    monkeypatch.setattr(exporter, "digest", tampered)
    with pytest.raises(exporter.ExportError, match="source changed"):
        run(request_data)
    assert_no_output(request_data)


def test_saved_file_tamper_blocks_publication(request_data, monkeypatch):
    original = exporter.write_csv
    paths = []
    def tamper(path, fields, rows):
        original(path, fields, rows); paths.append(path)
        if len(paths) == 4: paths[0].write_text("TAMPERED")
    monkeypatch.setattr(exporter, "write_csv", tamper)
    with pytest.raises(exporter.ExportError, match="output changed"):
        run(request_data)
    assert_no_output(request_data)


def test_shutdown_error_suppresses_manifest(request_data):
    backend = FakeBackend()
    def fail(): raise RuntimeError(SECRET)
    backend.before_shutdown = fail
    with pytest.raises(exporter.ExportError, match="withheld") as caught:
        run(request_data, backend)
    assert SECRET not in str(caught.value)
    assert_no_output(request_data)


def test_mac_cli_fails_before_import_or_connection(request_data, monkeypatch, capsys):
    monkeypatch.setattr(exporter.sys, "platform", "darwin")
    def forbidden_import(*_): raise AssertionError("MT5 import attempted on macOS")
    monkeypatch.setattr(exporter.importlib, "import_module", forbidden_import)
    code = exporter.main(["--terminal-path", request_data["terminal_path"], "--output", request_data["output"],
                          "--symbol", "BOOM500", "Boom 500 Index", "--window", *request_data["windows"][0]])
    assert code == 2 and "no connection attempted" in capsys.readouterr().err
    assert_no_output(request_data)


def test_nonrunning_terminal_does_not_initialize(request_data):
    backend = FakeBackend()
    with pytest.raises(exporter.ExportError, match="already be running"):
        exporter.export_quotes(request_data, backend, platform_name="win32", process_checker=lambda _: False,
                               environment={}, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert not backend.calls


def test_failed_initialize_still_detaches_without_secrets(request_data):
    backend = FakeBackend()
    def init(*args, **kwargs):
        backend.calls.append(("initialize", args, kwargs))
        return False
    backend.initialize = init
    with pytest.raises(exporter.ExportError, match="no login"):
        run(request_data, backend)
    assert [c[0] for c in backend.calls] == ["initialize", "shutdown"]
    assert_no_output(request_data)


def test_personal_fields_are_not_accessed(request_data):
    class PrivateAccount:
        trade_mode = 0
        server = "Deriv-Demo"
        company = "Deriv (SVG) LLC"
        def __getattr__(self, name): raise AssertionError("Private account field accessed: " + name)
    backend = FakeBackend(); backend.account = PrivateAccount()
    run(request_data, backend)


def test_disallowed_api_wrapper_rejects_before_backend():
    backend = FakeBackend()
    with pytest.raises(exporter.ExportError, match="Prohibited"):
        exporter.ReadOnlyCalls(backend).call("order_send", {})
    assert not backend.calls
