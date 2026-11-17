"""Command-line entry point for quick diagnostics and price-panel analysis."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._ffi import (
    LibraryNotFoundError,
    abi_version,
    get_lib,
    native_version,
)
from ._version import __version__
from . import config as model_config
from . import qflux
from ._native_backend import native_info as python_native_info
from .advice import trade_advice
from .analysis import analyze_prices
from .apps import portfolio_risk

_DATE_NAMES = {"date", "datetime", "time", "日期", "时间"}
_MISSING = {"", "na", "n/a", "nan", "null", "none", "-"}
_DATE_FORMATS = ("%Y%m%d", "%Y/%m/%d", "%d/%m/%Y")


@dataclass(frozen=True, slots=True)
class PriceFile:
    prices: np.ndarray
    names: list[str]
    dates: list[str] | None
    dropped_columns: list[str]
    warnings: list[str]


def _parse_date(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_number(value: str) -> float:
    text = value.strip()
    if text.lower() in _MISSING:
        return float("nan")
    return float(text)


def _resolve_date_column(header: Sequence[str], requested: str | None) -> int | None:
    if requested is not None:
        if requested in header:
            return list(header).index(requested)
        try:
            index = int(requested)
        except ValueError as exc:
            raise ValueError(f"date column {requested!r} was not found") from exc
        if not 0 <= index < len(header):
            raise ValueError(f"date column index {index} is out of range")
        return index
    if header and header[0].strip().lower() in _DATE_NAMES:
        return 0
    return None


def read_price_csv(
    path: str | Path,
    *,
    date_column: str | None = None,
    delimiter: str = ",",
) -> PriceFile:
    """Read a wide price CSV using only the standard library.

    The first row must be a header. A column named date/datetime/time/日期/时间
    is detected automatically and used as the row index; otherwise all columns
    are treated as assets. Rows are sorted when every date can be parsed, and
    duplicate dates are rejected. Unparseable non-empty numbers are errors,
    not silently skipped.
    """
    if not delimiter:
        raise ValueError("delimiter must not be empty")
    path = Path(path)
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"cannot open {path}: {exc}") from exc

    with handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = [value.strip() for value in next(reader)]
        except StopIteration as exc:
            raise ValueError("CSV is empty") from exc
        if not header or any(not value for value in header):
            raise ValueError("the CSV header must contain non-empty column names")
        if len(set(header)) != len(header):
            raise ValueError("the CSV header contains duplicate column names")

        date_index = _resolve_date_column(header, date_column)
        data_indices = [i for i in range(len(header)) if i != date_index]
        asset_names = [header[i] for i in data_indices]
        if not data_indices:
            raise ValueError("the CSV has no data columns")

        rows: list[list[str]] = []
        raw_dates: list[str] = []
        warnings: list[str] = []
        for line_number, row in enumerate(reader, start=2):
            values = [value.strip() for value in row]
            if not any(values):
                continue
            if len(values) != len(header):
                raise ValueError(
                    f"line {line_number} has {len(values)} fields, "
                    f"expected {len(header)}"
                )
            rows.append(values)
            if date_index is not None:
                raw_dates.append(values[date_index])

    if not rows:
        raise ValueError("the CSV has no data rows")

    numeric = np.full((len(rows), len(data_indices)), np.nan, dtype=np.float64)
    dropped_columns: list[str] = []
    for column, source_index in enumerate(data_indices):
        parsed = 0
        invalid: tuple[int, str] | None = None
        for row_number, row in enumerate(rows, start=2):
            text = row[source_index]
            if text.strip().lower() in _MISSING:
                continue
            try:
                numeric[row_number - 2, column] = _parse_number(text)
                parsed += 1
            except ValueError:
                if invalid is None:
                    invalid = (row_number, text)
        if parsed == 0:
            dropped_columns.append(header[source_index])
            continue
        if invalid is not None:
            row_number, text = invalid
            raise ValueError(
                f"line {row_number}, column {header[source_index]!r}: "
                f"{text!r} is not a number"
            )

    if dropped_columns:
        keep = [
            column
            for column, name in enumerate(asset_names)
            if name not in dropped_columns
        ]
        if not keep:
            raise ValueError("no numeric data columns were found")
        numeric = numeric[:, keep]
        asset_names = [asset_names[column] for column in keep]
        warnings.append(
            "dropped non-numeric columns: " + ", ".join(dropped_columns)
        )

    finite_rows = np.any(np.isfinite(numeric), axis=1)
    if not np.all(finite_rows):
        numeric = numeric[finite_rows]
        if date_index is not None:
            raw_dates = [
                value for value, keep in zip(raw_dates, finite_rows) if keep
            ]
        warnings.append(
            f"dropped {int((~finite_rows).sum())} rows with no numeric prices"
        )

    dates: list[str] | None = None
    if date_index is not None:
        if len(set(raw_dates)) != len(raw_dates):
            raise ValueError("the CSV contains duplicate dates")
        parsed_dates = [_parse_date(value) for value in raw_dates]
        if all(value is not None for value in parsed_dates):
            timestamps = [value for value in parsed_dates if value is not None]
            try:
                ordered = sorted(timestamps)
            except TypeError:
                ordered = timestamps
                warnings.append(
                    "date values mix timezone-aware and naive timestamps; "
                    "row order kept"
                )
            if timestamps != ordered:
                order = sorted(range(len(timestamps)), key=timestamps.__getitem__)
                numeric = numeric[order]
                raw_dates = [raw_dates[index] for index in order]
                warnings.append("rows were sorted by date")
            dates = raw_dates
        else:
            warnings.append("date column contains non-ISO values; row order kept")
            dates = raw_dates

    return PriceFile(
        prices=np.ascontiguousarray(numeric),
        names=asset_names,
        dates=dates,
        dropped_columns=dropped_columns,
        warnings=warnings,
    )


def _read_args_prices(args: argparse.Namespace) -> PriceFile:
    return read_price_csv(
        args.path,
        date_column=args.date_column,
        delimiter=args.delimiter,
    )


def _format_value(
    value: Any, *, percent: bool = False, width: int = 9, signed: bool = True
) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a".rjust(width)
    if percent:
        rendered = f"{value * 100:+.2f}%" if signed else f"{value * 100:.2f}%"
        return rendered.rjust(width)
    if isinstance(value, float):
        return f"{value:.3f}".rjust(width)
    return str(value).rjust(width)


def _add_date_metadata(report: dict[str, Any], parsed: PriceFile) -> None:
    if parsed.dates:
        report["input"]["date_start"] = parsed.dates[0]
        report["input"]["date_end"] = parsed.dates[-1]
    report["input"]["warnings"] = parsed.warnings


def _append_footer(lines: list[str], parsed: PriceFile) -> None:
    if parsed.dates:
        lines.append(f"date range: {parsed.dates[0]} .. {parsed.dates[-1]}")
    if parsed.warnings:
        lines.append("")
        lines.extend(f"warning: {message}" for message in parsed.warnings)


def _emit_output(output: str, destination: str | None) -> None:
    if destination:
        Path(destination).write_text(output + "\n", encoding="utf-8")
        print(f"wrote {destination}")
    else:
        print(output)


def _format_text(report: dict[str, Any], source: str, parsed: PriceFile) -> str:
    lines = [
        f"rflux {report['package']['version']} | "
        f"native {report['package']['native_version']} | "
        f"ABI {report['package']['abi_version']}",
        f"source: {source}",
        f"rows: {report['input']['observations']} | "
        f"assets: {report['input']['assets']}",
        "",
        f"{'asset':<18}{'ann.ret':>10}{'ann.vol':>10}{'Sharpe':>10}"
        f"{'maxDD':>10}{'RSI14':>9}",
    ]
    for name, stats in report["asset_stats"].items():
        lines.append(
            f"{name[:18]:<18}"
            f"{_format_value(stats.get('annualized_return'), percent=True)}"
            f"{_format_value(stats.get('annualized_volatility'), percent=True, signed=False)}"
            f"{_format_value(stats.get('sharpe'))}"
            f"{_format_value(stats.get('max_drawdown'), percent=True)}"
            f"{_format_value(stats.get('rsi_14'))}"
        )
    market = report["market"]
    lines.append(
        f"{'equal-weight':<18}"
        f"{_format_value(market.get('annualized_return'), percent=True)}"
        f"{_format_value(market.get('annualized_volatility'), percent=True, signed=False)}"
        f"{_format_value(market.get('sharpe'))}"
        f"{_format_value(market.get('max_drawdown'), percent=True)}"
        f"{_format_value(None)}"
    )
    correlation = report.get("correlation", {})
    if correlation:
        lines.extend(
            [
                "",
                "correlation: "
                f"min={_format_value(correlation.get('minimum')).strip()}, "
                f"mean={_format_value(correlation.get('mean')).strip()}, "
                f"max={_format_value(correlation.get('maximum')).strip()}",
            ]
        )
    network = report.get("network")
    if network:
        top = ", ".join(
            f"{item['asset']}={item['value']:.3f}"
            for item in network["top_eigenvector"][:3]
        )
        lines.append(
            "network: "
            f"edges={network['edges']}, "
            f"components={network['components']}, "
            f"largest={network['largest_component']}, "
            f"density={network['density']:.3f}"
        )
        if top:
            lines.append(f"top eigenvector centrality: {top}")
    hmm = report.get("hmm")
    if hmm:
        latest = ", ".join(f"{value:.3f}" for value in hmm["latest_filtered"])
        lines.append(
            f"HMM: states={hmm['states']}, latest={hmm['latest_state']}, "
            f"filtered=[{latest}], converged={hmm['converged']}"
        )
    _append_footer(lines, parsed)
    if report["errors"]:
        lines.append("")
        lines.extend(f"note: {message}" for message in report["errors"])
    return "\n".join(lines)


def _parse_weights(text: str) -> list[float]:
    values: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            raise ValueError("weights must be a comma-separated non-empty list")
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"invalid weight {item!r}") from exc
        if not math.isfinite(value):
            raise ValueError("weights must be finite")
        values.append(value)
    if not values:
        raise ValueError("weights must not be empty")
    return values


def _format_risk_text(report: dict[str, Any], source: str, parsed: PriceFile) -> str:
    lines = [
        f"rflux {report['package']['version']} | "
        f"source: {source}",
        f"rows: {report['input']['observations']} | "
        f"complete: {report['input']['complete_observations']} | "
        f"assets: {report['input']['asset_names']}",
        f"gross exposure: {report['input']['gross_exposure']:.3f} | "
        f"net exposure: {report['input']['net_exposure']:.3f}",
        "",
        f"{'asset':<18}{'weight':>10}{'ann.vol':>10}{'component':>11}{'share':>9}",
    ]
    for name, item in report["asset_risk"].items():
        lines.append(
            f"{name[:18]:<18}"
            f"{item['weight']:>10.3f}"
            f"{_format_value(item['annualized_volatility'], percent=True, signed=False)}"
            f"{_format_value(item['component_volatility'], percent=True, width=11)}"
            f"{_format_value(item['percent_contribution'], percent=True, width=9)}"
        )
    portfolio = report["portfolio"]
    lines.extend(
        [
            "",
            f"portfolio volatility: "
            f"{_format_value(portfolio['annualized_volatility'], percent=True, signed=False).strip()}",
            f"diversification ratio: "
            f"{_format_value(portfolio['diversification_ratio']).strip()}",
            f"gross HHI: {report['input']['gross_hhi']:.3f}",
        ]
    )
    metrics = portfolio["metrics"]
    lines.append(
        f"return: total={_format_value(metrics['total_return'], percent=True).strip()}, "
        f"ann.return={_format_value(metrics['annualized_return'], percent=True).strip()}, "
        f"Sharpe={_format_value(metrics['sharpe']).strip()}, "
        f"maxDD={_format_value(metrics['max_drawdown'], percent=True).strip()}"
    )
    _append_footer(lines, parsed)
    return "\n".join(lines)


def _command_risk(args: argparse.Namespace) -> int:
    parsed = _read_args_prices(args)
    weights = _parse_weights(args.weights)
    if len(weights) != len(parsed.names):
        raise ValueError(
            f"got {len(weights)} weights for {len(parsed.names)} assets: "
            f"{parsed.names}"
        )
    return_panel = qflux.returns_panel(parsed.prices)
    report = portfolio_risk(
        return_panel,
        weights,
        names=parsed.names,
        periods_per_year=args.periods_per_year,
        risk_free=args.risk_free,
        confidence=args.confidence,
    )
    _add_date_metadata(report, parsed)
    report["input"]["source"] = str(args.path)
    output = (
        json.dumps(report, ensure_ascii=False, indent=2)
        if args.format == "json"
        else _format_risk_text(report, str(args.path), parsed)
    )
    _emit_output(output, args.output)
    return 0


def _native_info() -> dict[str, Any]:
    try:
        lib = get_lib()
        return {
            "loaded": True,
            "version": native_version(),
            "abi_version": abi_version(),
            "path": getattr(lib, "_name", None),
        }
    except (LibraryNotFoundError, OSError, ImportError) as exc:
        return {"loaded": False, "error": str(exc)}


def _doctor_payload() -> tuple[dict[str, Any], bool]:
    native = _native_info()
    payload: dict[str, Any] = {
        "python_version": __version__,
        "native": native,
        "native_extension": python_native_info(),
        "optional": {},
    }
    for module in ("numpy", "pandas", "akshare", "yfinance"):
        payload["optional"][module] = importlib.util.find_spec(module) is not None
    return payload, bool(native["loaded"])


def _command_version(as_json: bool) -> int:
    native = _native_info()
    payload: dict[str, Any] = {"version": __version__}
    if native["loaded"]:
        payload["native_version"] = native["version"]
        payload["abi_version"] = native["abi_version"]
    else:
        payload["native_error"] = native["error"]
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"rflux {payload['version']}")
        if native["loaded"]:
            print(
                f"native library {payload['native_version']} "
                f"(ABI {payload['abi_version']})"
            )
        else:
            print(f"native library error: {payload['native_error']}")
    return 0 if native["loaded"] else 1


def _command_doctor(as_json: bool) -> int:
    payload, ok = _doctor_payload()
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        native = payload["native"]
        if native.get("loaded"):
            print("native library: loaded")
            print(f"  path: {native.get('path')}")
            print(f"  version: {native.get('version')}")
            print(f"  ABI: {native.get('abi_version')}")
        else:
            print("native library: failed to load")
            print(f"  {native.get('error')}")
        extension = payload.get("native_extension") or {}
        if extension.get("available"):
            print(
                "native extension: loaded "
                f"(stable ABI: {'yes' if extension.get('stable_abi') else 'no'}, "
                f"GIL release: {'yes' if extension.get('gil_released') else 'no'})"
            )
        else:
            print("native extension: not available (ctypes fallback active)")
        print("optional dependencies:")
        for module, present in payload["optional"].items():
            print(f"  {module}: {'yes' if present else 'no'}")
    return 0 if ok else 1


def _command_config(args: argparse.Namespace) -> int:
    loaded = model_config.load(args.path)
    model = loaded.model
    payload: dict[str, Any] = {
        "valid": True,
        "path": str(args.path),
        "model_type": loaded.type,
        "n_states": model.n_states,
        "n_features": model.n_features,
        "has_initial_parameters": model.initial is not None,
        "fit": model.fit.as_kwargs(),
        "native_extension": python_native_info(),
    }
    if args.build:
        instance = model.build()
        try:
            payload["constructed"] = True
            payload["ready"] = model.initial is not None
        finally:
            instance.close()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"config: {payload['path']}")
        print(f"  model: {payload['model_type']}")
        print(f"  states: {payload['n_states']}")
        print(f"  features: {payload['n_features']}")
        print(f"  initial parameters: {'yes' if payload['has_initial_parameters'] else 'no'}")
        print(f"  fit options: {json.dumps(payload['fit'], sort_keys=True)}")
        if args.build:
            print("  construction: ok")
    return 0


def _command_analyze(args: argparse.Namespace) -> int:
    parsed = _read_args_prices(args)
    report = analyze_prices(
        parsed.prices,
        names=parsed.names,
        periods_per_year=args.periods_per_year,
        risk_free=args.risk_free,
        include_hmm=not args.no_hmm,
        hmm_states=args.states,
        hmm_max_iter=args.max_iter,
        hmm_n_init=args.n_init,
        include_network=not args.no_network,
        network_threshold=args.threshold,
        seed=args.seed,
    )
    _add_date_metadata(report, parsed)
    report["input"]["dropped_columns"] = parsed.dropped_columns
    output = (
        json.dumps(report, ensure_ascii=False, indent=2)
        if args.format == "json"
        else _format_text(report, str(args.path), parsed)
    )
    _emit_output(output, args.output)
    return 0


def _format_advice_text(report: dict[str, Any], source: str, parsed: PriceFile) -> str:
    allocation = report["allocation"]
    lines = [
        f"rflux {report['package']['version']} | source: {source}",
        f"rows: {report['input']['observations']} | "
        f"assets: {report['input']['assets']}",
        f"exposure: {allocation['exposure']:.2f} | "
        f"cash: {allocation['cash_weight']:.2f} | "
        f"turnover: {allocation['expected_turnover']:.2f}",
        "",
        f"{'asset':<18}{'current':>10}{'target':>10}{'delta':>10}"
        f"{'score':>8}{'action':>10}",
    ]
    for item in allocation["actions"]:
        lines.append(
            f"{item['asset'][:18]:<18}"
            f"{item['current_weight']:>10.3f}"
            f"{item['target_weight']:>10.3f}"
            f"{item['delta']:>+10.3f}"
            f"{item['signal_score']:>+8.2f}"
            f"{item['action']:>10}"
        )
    lines.append("")
    lines.append("reasons:")
    for item in allocation["actions"]:
        detail = f"  {item['asset']}: {item['reason']}"
        if item.get("reference_stop") is not None:
            detail += f"; stop reference {item['reference_stop']:.2f}"
        if item.get("recommended_shares") is not None:
            detail += f"; shares {item['recommended_shares']}"
        lines.append(detail)
    risk = report["risk"]
    lines.append("")
    lines.append(
        f"market vol: {_format_value(risk['market_annualized_volatility'], percent=True, signed=False).strip()} | "
        f"target vol: {_format_value(risk['target_annualized_volatility'], percent=True, signed=False).strip()} | "
        f"target maxDD: {_format_value(risk['target_max_drawdown'], percent=True).strip()}"
    )
    if report.get("regime"):
        regime = report["regime"]
        lines.append(
            f"regime: state={regime['latest_state']} | "
            f"expected next return={regime['expected_next_return']:+.4f} | "
            f"stress={regime['stress_probability']:.2f} | "
            f"risk_off={regime['risk_off']}"
        )
    _append_footer(lines, parsed)
    if report["errors"]:
        lines.append("")
        lines.extend(f"note: {message}" for message in report["errors"])
    lines.append("")
    lines.append("notes:")
    lines.extend(f"  - {note}" for note in report["notes"])
    lines.append(report["disclaimer"])
    return "\n".join(lines)


def _command_advice(args: argparse.Namespace) -> int:
    parsed = _read_args_prices(args)
    weights = _parse_weights(args.weights) if args.weights else None
    if weights is not None and len(weights) != len(parsed.names):
        raise ValueError(
            f"got {len(weights)} weights for {len(parsed.names)} assets: "
            f"{parsed.names}"
        )
    report = trade_advice(
        parsed.prices,
        names=parsed.names,
        current_weights=weights,
        periods_per_year=args.periods_per_year,
        risk_free=args.risk_free,
        max_weight=args.max_weight,
        target_volatility=args.target_vol,
        confidence=args.confidence,
        portfolio_value=args.portfolio_value,
        risk_per_trade=args.risk_per_trade,
        include_hmm=not args.no_hmm,
        hmm_states=args.states,
        hmm_max_iter=args.max_iter,
        hmm_n_init=args.n_init,
        seed=args.seed,
    )
    _add_date_metadata(report, parsed)
    report["input"]["source"] = str(args.path)
    output = (
        json.dumps(report, ensure_ascii=False, indent=2)
        if args.format == "json"
        else _format_advice_text(report, str(args.path), parsed)
    )
    _emit_output(output, args.output)
    return 0


def _add_csv_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="wide CSV with a header")
    parser.add_argument("--date-column", default=None)
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", default=None)
    parser.add_argument("--periods-per-year", type=float, default=252.0)
    parser.add_argument("--risk-free", type=float, default=0.0)


def _add_hmm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-hmm", action="store_true")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rflux",
        description="Diagnostics and price-panel analysis for rflux.",
    )
    subparsers = parser.add_subparsers(dest="command")
    version = subparsers.add_parser("version", help="show Python/native versions")
    version.add_argument("--json", action="store_true")
    doctor = subparsers.add_parser("doctor", help="check native and optional dependencies")
    doctor.add_argument("--json", action="store_true")
    config_parser = subparsers.add_parser(
        "config", help="validate a TOML/JSON/YAML model configuration"
    )
    config_parser.add_argument("path", help="configuration file")
    config_parser.add_argument("--json", action="store_true")
    config_parser.add_argument(
        "--build", action="store_true", help="also instantiate the model"
    )
    analyze = subparsers.add_parser(
        "analyze", help="summary, network and HMM diagnostics for a price CSV"
    )
    _add_csv_arguments(analyze)
    _add_hmm_arguments(analyze)
    analyze.add_argument("--no-network", action="store_true")
    analyze.add_argument("--threshold", type=float, default=0.5)

    advice = subparsers.add_parser(
        "advice", help="risk-aware rebalance proposal for a price CSV"
    )
    _add_csv_arguments(advice)
    _add_hmm_arguments(advice)
    advice.add_argument("--weights", default=None,
                        help="current weights, comma-separated")
    advice.add_argument("--max-weight", type=float, default=0.35)
    advice.add_argument("--target-vol", type=float, default=0.15)
    advice.add_argument("--confidence", type=float, default=0.95)
    advice.add_argument("--portfolio-value", type=float, default=None)
    advice.add_argument("--risk-per-trade", type=float, default=0.005)

    risk = subparsers.add_parser(
        "risk", help="portfolio volatility and Euler risk contributions"
    )
    _add_csv_arguments(risk)
    risk.add_argument(
        "--weights", required=True,
        help="comma-separated weights in CSV data-column order"
    )
    risk.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "version":
            return _command_version(args.json)
        if args.command == "doctor":
            return _command_doctor(args.json)
        if args.command == "config":
            return _command_config(args)
        if args.command == "analyze":
            return _command_analyze(args)
        if args.command == "risk":
            return _command_risk(args)
        if args.command == "advice":
            return _command_advice(args)
        parser.error(f"unknown command {args.command}")
    except Exception as exc:  # keep the CLI usable without a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
