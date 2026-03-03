"""
models/repricing.py
-------------------
Repricing gap analysis and NII sensitivity for the NII/EVE simulator.

Design notes
------------
Bucket scheme
  This module uses a 7-bucket IRRBB-style ladder that is more granular than
  the 4-bucket summary inside ``balance_sheet.py``:

    <3M  |  3M-6M  |  6M-1Y  |  1Y-2Y  |  2Y-5Y  |  5Y-10Y  |  >10Y

  The ``build_repricing_gap()`` function also accepts custom boundaries
  (via ``buckets_yr``) to reproduce regulatory or internal reporting schemes.

Repricing convention
  *Fixed-rate* instruments reprice once at maturity.  Their notional is
  placed in the bucket whose range contains ``repricing_tenor_months`` (which
  equals their maturity in months).

  *Floating-rate* instruments reprice every ``repricing_tenor_months`` months.
  Their notional is placed in the bucket corresponding to that reset frequency.

NII methodology
  NII impact is calculated over a 12-month horizon.  Two shock scenarios are
  supported:

  ``"instant"``
      The full shock is applied at t = 0.  Each instrument's rate is updated
      at its next repricing event (midpoint of current period on average) and
      earns the shocked rate for the remainder of the horizon.

  ``"ramp"``
      The shock builds linearly from 0 bp at t = 0 to ``shock_bp`` bp at
      t = horizon.  At each repricing event the instrument locks in the
      prevailing (partial) shock for the period until the next event.

  The **uniform-distribution assumption** is applied to floating instruments:
  a 3M floater is equally likely to be anywhere in its reset cycle, so the
  expected time to next reset = R / 2 months.  Subsequent resets follow at
  R-month intervals.

  Only instruments whose next repricing falls *strictly before* the end of the
  horizon contribute to ΔNII.

All monetary values are in $M.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from models.balance_sheet import BalanceSheet, Instrument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bucket definitions
# ---------------------------------------------------------------------------

# Each tuple: (label, lower_months inclusive, upper_months exclusive, midpoint_months)
_BUCKET_SPECS: list[tuple[str, float, float, float]] = [
    ("<3M",     0.0,    3.0,   1.5),
    ("3M-6M",   3.0,    6.0,   4.5),
    ("6M-1Y",   6.0,   12.0,   9.0),
    ("1Y-2Y",  12.0,   24.0,  18.0),
    ("2Y-5Y",  24.0,   60.0,  42.0),
    ("5Y-10Y", 60.0,  120.0,  90.0),
    (">10Y",  120.0,  float("inf"), 180.0),
]

BUCKET_LABELS: list[str]        = [s[0] for s in _BUCKET_SPECS]
BUCKET_MIDPOINTS: dict[str, float] = {s[0]: s[3] for s in _BUCKET_SPECS}  # months

NII_HORIZON_MONTHS: int = 12   # standard 1-year NII measurement window

# Type alias published for downstream consumers (e.g. risk_metrics.py)
GapLadder = pd.DataFrame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign_bucket(
    repricing_tenor_months: float,
    specs: list[tuple[str, float, float, float]] = _BUCKET_SPECS,
) -> str:
    """Return the bucket label for a given repricing tenor (in months)."""
    for label, lo, hi, _ in specs:
        if lo <= repricing_tenor_months < hi:
            return label
    return specs[-1][0]  # overflow to last bucket


def _build_custom_bucket_specs(
    buckets_yr: list[float],
) -> list[tuple[str, float, float, float]]:
    """
    Convert a list of upper-boundary tenors in years into bucket specs.

    Parameters
    ----------
    buckets_yr :
        Sorted upper-bound tenors in years, e.g. ``[0.25, 0.5, 1, 2, 5, 10]``.
        An open-ended final bucket (``>last``) is appended automatically.

    Returns
    -------
    list of (label, lower_months, upper_months, midpoint_months)
    """
    bounds = sorted(b * 12 for b in buckets_yr)
    specs: list[tuple[str, float, float, float]] = []
    prev = 0.0
    for hi in bounds:
        lo = prev
        mid = (lo + hi) / 2.0
        label = _month_label(lo, hi)
        specs.append((label, lo, hi, mid))
        prev = hi
    # Open-ended tail bucket
    specs.append((f">{prev:.0f}M", prev, float("inf"), prev * 2.0))
    return specs


def _month_label(lo: float, hi: float) -> str:
    """Format a bucket label from month boundaries (e.g. '3M-6M', '<3M', '>120M')."""
    if lo == 0:
        return f"<{hi:.0f}M"
    if hi == float("inf"):
        return f">{lo:.0f}M"
    return f"{lo:.0f}M-{hi:.0f}M"


def _repricing_events(
    instr: Instrument,
    horizon_months: int,
) -> list[float]:
    """
    Return the list of repricing times (months from now) within [0, horizon_months).

    Convention
    ----------
    - **Fixed**: single event at ``repricing_tenor_months`` (= maturity for fixed
      instruments).  Excluded if at or beyond the horizon.
    - **Floating**: first event at ``R / 2`` (uniform-distribution assumption),
      then every ``R`` months, stopping at ``min(horizon_months, maturity)``.

    Parameters
    ----------
    instr :
        The instrument to analyse.
    horizon_months :
        NII measurement window in months.

    Returns
    -------
    list[float]
        Sorted event times in months; empty if no repricing within the horizon.
    """
    R = instr.repricing_tenor_months

    # Effective maturity cap for floating instruments
    if instr.maturity_years is not None:
        mat_m = instr.maturity_years * 12.0
    elif instr.behavioural_maturity_years is not None:
        mat_m = instr.behavioural_maturity_years * 12.0
    else:
        mat_m = float("inf")

    if instr.rate_type == "fixed":
        # Single repricing event at maturity (= R for fixed in our model)
        return [R] if R < horizon_months else []

    # Floating: periodic resets starting at R/2 (mid-cycle uniform assumption)
    R = max(R, 0.01)   # guard against zero / near-zero
    t_first = R / 2.0
    cap = min(float(horizon_months), mat_m)
    events: list[float] = []
    t = t_first
    while t < cap:
        events.append(t)
        t += R
    return events


def _instrument_nii_delta(
    instr: Instrument,
    shock_decimal: float,
    scenario: str,
    horizon_months: int,
) -> float:
    """
    Calculate the NII delta for a single instrument under a rate shock.

    Parameters
    ----------
    instr :
        The instrument being evaluated.
    shock_decimal :
        Rate shock expressed as a decimal (e.g. ``0.01`` for +100 bp).
    scenario :
        ``"instant"`` — full shock applied at t = 0; each item reprices at
        its next reset and earns the shocked rate for the rest of the horizon.

        ``"ramp"``    — shock ramps linearly from 0 at t = 0 to
        ``shock_decimal`` at t = ``horizon_months``; each item locks in the
        prevailing (partial) shock at its reset date.
    horizon_months :
        Length of the NII horizon in months.

    Returns
    -------
    float
        ΔNII in $M.  Positive = income improvement; negative = deterioration.
        Sign convention: asset income gains are positive; liability cost
        increases reduce NII (negative).
    """
    events = _repricing_events(instr, horizon_months)
    if not events:
        return 0.0

    H    = float(horizon_months)
    sign = 1.0 if instr.side == "asset" else -1.0

    if scenario == "instant":
        # Full shock is locked in at the first repricing event and held for
        # all subsequent resets within the horizon (same shock level applies).
        t_first = events[0]
        delta = instr.notional * shock_decimal * (H - t_first) / H

    else:  # ramp
        # At each reset event t_k, the prevailing shock fraction is t_k / H.
        # The instrument earns that rate until the next event (or horizon end).
        delta = 0.0
        for k, t_k in enumerate(events):
            t_next      = events[k + 1] if k + 1 < len(events) else H
            hold_frac   = (min(t_next, H) - t_k) / H   # fraction of year
            shock_here  = shock_decimal * (t_k / H)
            delta      += instr.notional * shock_here * hold_frac

    return sign * delta


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NiiResult:
    """
    Complete NII sensitivity result for one shock/scenario combination.

    Attributes
    ----------
    shock_bp :
        Applied parallel rate shock in basis points.
    scenario :
        ``"instant"`` or ``"ramp"``.
    base_nii :
        Annualised NII at current rates ($M).
    total_delta_nii :
        Change in NII due to the shock ($M).
    by_instrument :
        Per-instrument breakdown — columns: ``side``, ``name``, ``notional``,
        ``rate_type``, ``repricing_tenor_months``, ``bucket``,
        ``reprices_in_horizon``, ``delta_nii``, ``pct_of_total_delta``.
    by_bucket :
        NII impact aggregated over the 7-bucket ladder — columns: ``bucket``,
        ``midpoint_months``, ``reprices_in_horizon``, ``delta_nii``,
        ``asset_delta``, ``liability_delta``.
    horizon_months :
        NII measurement window in months.
    """

    shock_bp:        float
    scenario:        str
    base_nii:        float
    total_delta_nii: float
    by_instrument:   pd.DataFrame = field(repr=False)
    by_bucket:       pd.DataFrame = field(repr=False)
    horizon_months:  int = NII_HORIZON_MONTHS

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def shocked_nii(self) -> float:
        """NII under the shocked rate environment ($M)."""
        return self.base_nii + self.total_delta_nii

    @property
    def pct_change(self) -> float:
        """Percentage change in NII relative to the base."""
        if abs(self.base_nii) < 1e-9:
            return 0.0
        return self.total_delta_nii / abs(self.base_nii) * 100

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self) -> None:
        """Pretty-print the full NII sensitivity report to stdout."""
        SEP  = "=" * 95
        DASH = "-" * 95
        scen_label = (
            f"Ramp (linear over {self.horizon_months} months)"
            if self.scenario == "ramp"
            else "Instantaneous (t = 0)"
        )

        print(SEP)
        print(
            f"  NII SENSITIVITY  |  Shock: {self.shock_bp:+.0f} bp  "
            f"|  Scenario: {scen_label}  |  Horizon: {self.horizon_months}M"
        )
        print(DASH)
        print(f"  {'Base NII':40s}  {self.base_nii:>10.3f}  $M")
        print(f"  {'Delta NII':40s}  {self.total_delta_nii:>+10.3f}  $M")
        print(f"  {'Shocked NII':40s}  {self.shocked_nii:>10.3f}  $M")
        print(f"  {'NII % change':40s}  {self.pct_change:>+10.2f}  %")

        print(f"\n  BY REPRICING BUCKET")
        print(DASH)
        with pd.option_context("display.float_format", "{:,.4f}".format):
            print(self.by_bucket.to_string(index=False))

        print(f"\n  BY INSTRUMENT")
        print(DASH)
        disp = self.by_instrument[[
            "side", "name", "notional", "repricing_tenor_months",
            "rate_type", "reprices_in_horizon", "delta_nii", "pct_of_total_delta",
        ]].copy()
        with pd.option_context(
            "display.float_format", "{:,.4f}".format,
            "display.max_colwidth", 30,
        ):
            print(disp.to_string(index=False))
        print(SEP)


# ---------------------------------------------------------------------------
# Public API – repricing gap ladder
# ---------------------------------------------------------------------------

def build_repricing_gap(
    balance_sheet: BalanceSheet,
    buckets_yr: list[float] | None = None,
) -> GapLadder:
    """
    Build a repricing gap ladder.

    Each instrument is assigned to the bucket whose range contains its
    ``repricing_tenor_months``.  For fixed instruments this is their
    maturity; for floating instruments it is their reset frequency.
    The gap is ``assets − liabilities`` within each bucket.

    Parameters
    ----------
    balance_sheet :
        Populated balance sheet instance.
    buckets_yr :
        Custom bucket upper-boundary tenors in years.  When *None* the
        default 7-bucket IRRBB ladder is used.

    Returns
    -------
    GapLadder (pd.DataFrame)
        Columns:

        ==================== =================================================
        ``bucket``           Bucket label
        ``lower_months``     Lower boundary (months, inclusive)
        ``upper_months``     Upper boundary (months, exclusive)
        ``midpoint_months``  Representative midpoint in months
        ``asset_notional``   Sum of repricing asset notionals in bucket ($M)
        ``asset_pct``        As % of total assets
        ``liability_notional``  Sum of repricing liability notionals ($M)
        ``liability_pct``    As % of total liabilities
        ``gap``              ``asset_notional − liability_notional``
        ``cumulative_gap``   Running sum of ``gap`` from first bucket
        ==================== =================================================
    """
    specs = _build_custom_bucket_specs(buckets_yr) if buckets_yr else _BUCKET_SPECS
    labels = [s[0] for s in specs]

    asset_by_bucket: dict[str, float] = {lb: 0.0 for lb in labels}
    liab_by_bucket:  dict[str, float] = {lb: 0.0 for lb in labels}

    for instr in balance_sheet.assets:
        b = _assign_bucket(instr.repricing_tenor_months, specs)
        asset_by_bucket[b] += instr.notional

    for instr in balance_sheet.liabilities:
        b = _assign_bucket(instr.repricing_tenor_months, specs)
        liab_by_bucket[b] += instr.notional

    ta = balance_sheet.total_assets      or 1.0
    tl = balance_sheet.total_liabilities or 1.0

    rows = []
    for label, lo, hi, mid in specs:
        a = asset_by_bucket[label]
        l = liab_by_bucket[label]
        rows.append({
            "bucket":             label,
            "lower_months":       lo,
            "upper_months":       hi if hi != float("inf") else None,
            "midpoint_months":    mid,
            "asset_notional":     a,
            "asset_pct":          round(a / ta * 100, 1),
            "liability_notional": l,
            "liability_pct":      round(l / tl * 100, 1),
            "gap":                a - l,
        })

    df = pd.DataFrame(rows)
    df["cumulative_gap"] = df["gap"].cumsum()

    logger.info(
        "Repricing gap ladder: %d buckets  |  net gap = %.1f  |  "
        "total assets repriced = %.1f  |  total liabilities repriced = %.1f",
        len(labels),
        df["gap"].sum(),
        df["asset_notional"].sum(),
        df["liability_notional"].sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Public API – NII sensitivity
# ---------------------------------------------------------------------------

def compute_nii_impact(
    balance_sheet: BalanceSheet,
    shock_bp: float,
    scenario: str = "instant",
    horizon_months: int = NII_HORIZON_MONTHS,
) -> NiiResult:
    """
    Compute NII sensitivity to a parallel rate shock over a 12-month horizon.

    Parameters
    ----------
    balance_sheet :
        Populated balance sheet instance.
    shock_bp :
        Parallel rate shock in basis points (positive = rates up).
    scenario :
        ``"instant"`` — shock is applied at t = 0 (stress / worst-case).
        ``"ramp"``    — shock builds linearly from 0 to ``shock_bp`` over
        ``horizon_months`` (gradual normalisation scenario).
    horizon_months :
        NII horizon in months (default 12).

    Returns
    -------
    NiiResult
        Structured container with total ΔNII, per-instrument and per-bucket
        breakdowns, and summary properties (``shocked_nii``, ``pct_change``).

    Notes
    -----
    Only instruments with a repricing event *strictly before* ``horizon_months``
    contribute to ΔNII.  Fixed instruments with maturity >= horizon are
    excluded; floating instruments whose first reset falls at or after the
    horizon are excluded.

    The sign convention follows banking practice:

    * A rate increase on an **asset** raises income → positive ΔNII.
    * A rate increase on a **liability** raises cost → negative ΔNII.
    * Total ΔNII = Σ(asset contributions) + Σ(liability contributions).
    """
    if scenario not in ("instant", "ramp"):
        raise ValueError(f"scenario must be 'instant' or 'ramp', got {scenario!r}")

    shock_decimal = shock_bp / 10_000.0
    base_nii      = balance_sheet.net_interest_income

    # --- Per-instrument calculation ---
    instr_rows: list[dict] = []
    for instr in balance_sheet.assets + balance_sheet.liabilities:
        delta  = _instrument_nii_delta(instr, shock_decimal, scenario, horizon_months)
        events = _repricing_events(instr, horizon_months)
        instr_rows.append({
            "side":                    instr.side,
            "name":                    instr.name,
            "notional":                instr.notional,
            "rate_type":               instr.rate_type,
            "coupon_pct":              round(instr.coupon * 100, 3),
            "repricing_tenor_months":  instr.repricing_tenor_months,
            "bucket":                  _assign_bucket(instr.repricing_tenor_months),
            "reprices_in_horizon":     len(events) > 0,
            "n_reprice_events":        len(events),
            "delta_nii":               round(delta, 6),
        })

    by_instrument = pd.DataFrame(instr_rows)
    total_delta   = float(by_instrument["delta_nii"].sum())

    # Percentage-of-total column (handle near-zero total gracefully)
    if abs(total_delta) > 1e-9:
        by_instrument["pct_of_total_delta"] = (
            by_instrument["delta_nii"] / total_delta * 100
        ).round(1)
    else:
        by_instrument["pct_of_total_delta"] = 0.0

    # --- Per-bucket aggregation ---
    # Split asset and liability contributions before summing
    asset_delta_by_bucket = (
        by_instrument[by_instrument["side"] == "asset"]
        .groupby("bucket", sort=False)["delta_nii"]
        .sum()
        .reindex(BUCKET_LABELS, fill_value=0.0)
    )
    liab_delta_by_bucket = (
        by_instrument[by_instrument["side"] == "liability"]
        .groupby("bucket", sort=False)["delta_nii"]
        .sum()
        .reindex(BUCKET_LABELS, fill_value=0.0)
    )

    by_bucket = pd.DataFrame({
        "bucket":              BUCKET_LABELS,
        "midpoint_months":     [BUCKET_MIDPOINTS[b] for b in BUCKET_LABELS],
        "reprices_in_horizon": [BUCKET_MIDPOINTS[b] < horizon_months for b in BUCKET_LABELS],
        "asset_delta":         asset_delta_by_bucket.values,
        "liability_delta":     liab_delta_by_bucket.values,
        "delta_nii":           (asset_delta_by_bucket + liab_delta_by_bucket).values,
    })
    by_bucket["cumulative_delta_nii"] = by_bucket["delta_nii"].cumsum()

    logger.info(
        "NII impact | scenario=%s  shock=%+.0fbp  base=%.2f  delta=%+.3f  "
        "shocked=%.2f  change=%+.2f%%",
        scenario, shock_bp, base_nii, total_delta,
        base_nii + total_delta,
        (total_delta / abs(base_nii) * 100) if base_nii else 0.0,
    )

    return NiiResult(
        shock_bp=shock_bp,
        scenario=scenario,
        base_nii=round(base_nii, 4),
        total_delta_nii=round(total_delta, 6),
        by_instrument=by_instrument,
        by_bucket=by_bucket,
        horizon_months=horizon_months,
    )


def compare_nii_scenarios(
    balance_sheet: BalanceSheet,
    shock_bp: float,
    horizon_months: int = NII_HORIZON_MONTHS,
) -> pd.DataFrame:
    """
    Compare instantaneous and ramp NII impacts side-by-side for a given shock.

    Parameters
    ----------
    balance_sheet :
        Populated balance sheet.
    shock_bp :
        Rate shock in basis points.
    horizon_months :
        NII horizon in months (default 12).

    Returns
    -------
    pd.DataFrame
        Columns: ``scenario``, ``shock_bp``, ``base_nii``, ``delta_nii``,
        ``shocked_nii``, ``pct_change``.
    """
    instant = compute_nii_impact(balance_sheet, shock_bp, "instant", horizon_months)
    ramp    = compute_nii_impact(balance_sheet, shock_bp, "ramp",    horizon_months)

    rows = [
        {
            "scenario":    r.scenario,
            "shock_bp":    r.shock_bp,
            "base_nii":    r.base_nii,
            "delta_nii":   r.total_delta_nii,
            "shocked_nii": r.shocked_nii,
            "pct_change":  round(r.pct_change, 3),
        }
        for r in (instant, ramp)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public API – gap duration
# ---------------------------------------------------------------------------

def gap_duration(gap_ladder: GapLadder) -> float:
    """
    Compute the notional-weighted average repricing duration of the gap.

    The metric measures where in the maturity spectrum the repricing gaps
    are concentrated:

    * **Positive** value: gap-weighted repricing is skewed to long tenors
      (long-dated fixed assets dominate).
    * **Negative** value: would arise if longer-dated liabilities outweigh
      long-dated assets (uncommon in practice).

    Formula::

        gap_duration = Σ(|gap_i| × midpoint_yr_i × sign(gap_i)) / Σ(|gap_i|)

    Parameters
    ----------
    gap_ladder :
        Output of :func:`build_repricing_gap`.

    Returns
    -------
    float
        Signed gap duration in years.
    """
    if gap_ladder.empty:
        return 0.0

    midpoints_yr = gap_ladder["midpoint_months"] / 12.0
    gaps         = gap_ladder["gap"]
    abs_gaps     = gaps.abs()
    total_abs    = float(abs_gaps.sum())

    if total_abs < 1e-9:
        logger.warning("All gaps are zero; gap_duration returns 0.")
        return 0.0

    # Signed: each bucket contributes positively (asset-heavy) or negatively
    signed_contributions = midpoints_yr * gaps   # preserves sign via gap sign
    duration = float(signed_contributions.sum()) / total_abs

    logger.info(
        "Gap duration: %.3f years  (net gap: %.1f,  total |gap|: %.1f)",
        duration, float(gaps.sum()), total_abs,
    )
    return duration


# ---------------------------------------------------------------------------
# Convenience display
# ---------------------------------------------------------------------------

def display_gap_ladder(gap_ladder: GapLadder) -> None:
    """Pretty-print a repricing gap ladder to stdout."""
    SEP  = "=" * 95
    DASH = "-" * 95

    net_gap = gap_ladder["gap"].sum()
    gd      = gap_duration(gap_ladder)

    print(SEP)
    print("  REPRICING GAP LADDER (7 buckets)")
    print(DASH)

    display_cols = [
        "bucket", "midpoint_months",
        "asset_notional", "asset_pct",
        "liability_notional", "liability_pct",
        "gap", "cumulative_gap",
    ]
    available = [c for c in display_cols if c in gap_ladder.columns]
    with pd.option_context("display.float_format", "{:,.2f}".format):
        print(gap_ladder[available].to_string(index=False))

    print(DASH)
    print(f"  {'Net repricing gap':35s}  {net_gap:>10.1f}  $M")
    print(f"  {'Gap duration':35s}  {gd:>10.3f}  years")
    print(SEP)
