"""Notebook-parity fits of signed differential arrivals and distances."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

from .sources import NS_PER_SECOND, SourceCatalog

if TYPE_CHECKING:
    from .review import Blow, Pick, Receiver, ReviewState, Role


METHOD_VERSION = "hammer-differential-ols-v1"


def in_strict_lag_window(lag_ns: int) -> bool:
    """Test the legacy asymmetric, exclusive bounds on an imported signed lag."""
    return -60_000_000 < lag_ns < 80_000_000


@dataclass(frozen=True)
class Observation:
    blow_id: str
    lag_ns: int | None
    dx_m: float | None
    status: str
    reason: str
    requested_included: bool
    effective_included: bool
    residual_m: float | None = None
    original_mapping_offset_ns: int | None = None
    calibration_mapping_offset_ns: int | None = None

    @property
    def lag_s(self) -> float | None:
        return None if self.lag_ns is None else self.lag_ns / NS_PER_SECOND


@dataclass(frozen=True)
class Fit:
    observations: tuple[Observation, ...]
    blow_ids: tuple[str, ...]
    status: str
    diagnostic: str
    rank: int | None
    residual_df: int | None = None
    slope_m_per_s: float | None = None
    intercept_m: float | None = None


@dataclass(frozen=True)
class Analysis:
    baseline_all: Fit
    baseline_strict: Fit
    reviewed: Fit
    field_velocity_m_per_s: float | None
    field_interpretation: str


def _reviewed_time(
    catalog: SourceCatalog, state: ReviewState, pick: Pick, receiver: Receiver | None
) -> tuple[int | None, str, str]:
    ref = pick.reviewed_ref
    if ref is None:
        if pick.imported_ns is None:
            return None, "missing", "The arrival pick is missing."
        if receiver is not None:
            lookup = catalog.lookup(
                receiver.channel, pick.imported_ns, source_hash=receiver.source_hash or None
            )
            if lookup.status == "ambiguous":
                return None, "ambiguous", "The arrival overlaps multiple source segments."
            if lookup.status == "gap":
                return None, "missing", "The imported arrival has no recorded sample coverage."
        return None, "unresolved", "The arrival has no selected source sample."
    if receiver is not None and (
        ref.channel != receiver.channel
        or (receiver.source_hash and ref.source_hash != receiver.source_hash)
    ):
        return None, "unresolved", "The selected sample belongs to a different receiver."
    try:
        time_ns = catalog.time_ns(ref)
    except ValueError:
        return None, "unresolved", "The selected source sample is unavailable."
    if state.is_masked(ref):
        return None, "masked", "The selected sample is in an excluded interval."
    return time_ns, "valid", ""


def _mapping_offset(catalog: SourceCatalog, pick: Pick) -> int | None:
    if pick.original_ref is None or pick.imported_ns is None:
        return None
    try:
        return catalog.time_ns(pick.original_ref) - pick.imported_ns
    except ValueError:
        return None


def _observation(
    blow: Blow,
    requested: bool,
    original_ns: int | None,
    calibration_ns: int | None,
    sign: int,
    status: str = "valid",
    reason: str = "",
) -> Observation:
    x1, x2 = blow.distance_original_m, blow.distance_calibration_m
    dx_m = None
    if x1 is not None and x2 is not None and all(isfinite(value) and value >= 0 for value in (x1, x2)):
        dx_m = sign * (x1 - x2)
    lag_ns = None
    if blow.pair_status != "confirmed":
        status = "ambiguous" if blow.pair_status == "ambiguous" else "missing"
        reason = blow.pair_reason or "The receiver arrivals have no confirmed blow association."
    elif status == "valid":
        if original_ns is None or calibration_ns is None:
            status, reason = "missing", "A receiver arrival is missing."
        else:
            lag_ns = sign * (original_ns - calibration_ns)
            if dx_m is None:
                status, reason = "invalid_geometry", "Both receiver distances must be finite and nonnegative."
            elif not requested:
                status, reason = "excluded", "" if blow.reason else "Omitted from this fit."
    details = " ".join(dict.fromkeys(text for text in (reason, blow.reason) if text))
    return Observation(blow.blow_id, lag_ns, dx_m, status, details, requested, status == "valid")


def _reviewed_observation(
    catalog: SourceCatalog,
    state: ReviewState,
    blow: Blow,
    imported: Blow,
    receivers: Mapping[Role, Receiver],
    sign: int,
) -> Observation:
    original_ns, status, reason = _reviewed_time(
        catalog, state, blow.original, receivers.get("original")
    )
    calibration_ns, calibration_status, calibration_reason = _reviewed_time(
        catalog, state, blow.calibration, receivers.get("calibration")
    )
    if status != "valid":
        reason = f"Original receiver. {reason}"
    elif calibration_status != "valid":
        status, reason = calibration_status, f"Calibration receiver. {calibration_reason}"
    observed = _observation(blow, blow.included, original_ns, calibration_ns, sign, status, reason)
    return replace(
        observed,
        original_mapping_offset_ns=_mapping_offset(catalog, imported.original),
        calibration_mapping_offset_ns=_mapping_offset(catalog, imported.calibration),
    )


def _fit(rows: tuple[Observation, ...]) -> Fit:
    selected = [
        (row.blow_id, row.lag_ns, row.dx_m)
        for row in rows
        if row.effective_included and row.lag_ns is not None and row.dx_m is not None
    ]
    ids = tuple(blow_id for blow_id, _, _ in selected)
    if len(selected) < 2:
        return Fit(
            rows, ids, "insufficient_support", "At least two valid included pairs are required.", len(selected)
        )
    if len({lag_ns for _, lag_ns, _ in selected}) == 1:
        return Fit(rows, ids, "rank_deficient", "Included pairs all have the same lag.", 1)

    lags = [lag_ns / NS_PER_SECOND for _, lag_ns, _ in selected]
    distances = [dx_m for _, _, dx_m in selected]
    try:
        # Cell 11's degree-one least-squares fit with an intercept. Full output supplies the rank.
        with np.errstate(over="ignore", invalid="ignore"):
            coefficients, _, raw_rank, _, _ = np.polyfit(lags, distances, 1, full=True)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
        return Fit(rows, ids, "nonfinite", "Least squares could not compute finite coefficients.", None)
    rank = int(raw_rank)
    if rank != 2:
        return Fit(rows, ids, "rank_deficient", "The included lags cannot resolve a slope and intercept.", rank)
    slope, intercept = map(float, coefficients)
    df = len(selected) - rank
    if not (isfinite(slope) and isfinite(intercept)):
        return Fit(rows, ids, "nonfinite", "Least squares produced nonfinite coefficients.", rank, df)

    status = "two_point" if df == 0 else "ok"
    diagnostic = (
        "Two distinct lags define a line with zero residual degrees of freedom."
        if df == 0
        else f"{len(selected)} included pairs and {df} residual degrees of freedom."
    )
    if len(set(distances)) == 1:
        diagnostic += " Included distance differences are identical."
    with_residuals = []
    for row in rows:
        if row.lag_ns is not None and row.dx_m is not None:
            residual = row.dx_m - (slope * (row.lag_ns / NS_PER_SECOND) + intercept)
            if isfinite(residual):
                row = replace(row, residual_m=residual)
            else:
                row = replace(row, reason=f"{row.reason} The residual is not finite.".strip())
        with_residuals.append(row)
    return Fit(tuple(with_residuals), ids, status, diagnostic, rank, df, slope, intercept)


def analyze(
    catalog: SourceCatalog, initial: ReviewState, applied: ReviewState, *, reverse: bool = False
) -> Analysis:
    """Fit original imported observations and the currently applied sample references.

    The strict baseline uses imported lags in the original receiver order.
    Its membership is fixed across review edits and sign reversal.
    Mapping offsets report selected original sample minus imported UTC by receiver.
    """
    sign = -1 if reverse else 1
    imported = {blow.blow_id: blow for blow in initial.blows}
    receivers = {receiver.role: receiver for receiver in applied.receivers}
    all_rows = []
    strict_rows = []
    for blow in initial.blows:
        original_ns, calibration_ns = blow.original.imported_ns, blow.calibration.imported_ns
        strict = (
            original_ns is not None
            and calibration_ns is not None
            and in_strict_lag_window(original_ns - calibration_ns)
        )
        all_rows.append(_observation(blow, True, original_ns, calibration_ns, sign))
        strict_rows.append(_observation(blow, strict, original_ns, calibration_ns, sign))
    reviewed = _fit(tuple(
        _reviewed_observation(catalog, applied, blow, imported.get(blow.blow_id, blow), receivers, sign)
        for blow in applied.blows
    ))
    return Analysis(
        _fit(tuple(all_rows)),
        _fit(tuple(strict_rows)),
        reviewed,
        None,
        "Field velocity withheld. Same-blow phase pairing, receiver clocks, and geometry require field validation.",
    )
