"""
models/risk_metrics.py
----------------------
EVE (Economic Value of Equity) and NII risk metric calculations.

Design notes
------------
Yield curve
  All yield curves are stored as ``pd.Series`` with float tenor (years) as
  index and rates expressed as **decimals** (0.035 = 3.5 %).  The public
  ``load_base_curve()`` helper converts the % values returned by
  ``data.fred_loader`` automatically.

Cash flow convention
  Annual coupon payments are used throughout for simplicity.  Fractional
  final periods receive a prorated coupon plus the full principal.

  *Fixed-rate* instruments generate coupons to maturity and a principal at
  maturity.

  *Floating-rate* instruments use the "repricing-at-par" assumption: cash
  flows are generated only to the next repricing event; at that point the
  full principal is returned at par (the instrument is assumed to refinance
  at the then-prevailing market rate).

Discounting
  Spot rates are interpolated linearly from the yield curve at each cash
  flow date.  Annual compounding:  D(t) = 1 / (1 + r(t))^t.
  A 1 bp floor is applied to all rates to prevent numerical issues in
  extreme downside scenarios.

NII methodology
  ``project_nii`` uses the full shocked yield curve to determine the
  rate change at each instrument's repricing tenor, giving an accurate
  per-instrument NII delta for non-parallel shock scenarios (twist, bear
  flattener, etc.).

BP01
  Central-difference: BP01 = (EVE(base+1bp) − EVE(base−1bp)) / 2.
  Computed on the base curve so it represents the current-state sensitivity.

Duration gap
  DG = DA − (PV_Liabilities / PV_Assets) × DL
  where DA and DL are PV-weighted average modified durations.

All monetary values are in $M.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from models.balance_sheet import BalanceSheet, Instrument
from scenarios.rate_scenarios import RateScenario

logger = logging.getLogger(__name__)

_RATE_FLOOR: float = 0.0001   # 1 bp — prevents log/division errors in extreme shocks


# ---------------------------------------------------------------------------
# Yield curve utilities
# ---------------------------------------------------------------------------

def _interpolate_rate(curve: pd.Series, tenor_yr: float) -> float:
    """
    Linearly interpolate (flat-extrapolate) a yield curve at ``tenor_yr``.

    Parameters
    ----------
    curve :
        Tenor-indexed series (index = years, values = decimal rates).
    tenor_yr :
        Tenor to look up in years.

    Returns
    -------
    float
        Interpolated rate (decimal), floored at ``_RATE_FLOOR``.
    """
    if curve.empty:
        raise ValueError("Yield curve is empty — cannot interpolate.")

    tenors = np.asarray(curve.index, dtype=float)
    rates  = np.asarray(curve.values, dtype=float)

    if tenor_yr <= tenors[0]:
        rate = float(rates[0])
    elif tenor_yr >= tenors[-1]:
        rate = float(rates[-1])
    else:
        rate = float(np.interp(tenor_yr, tenors, rates))

    return max(rate, _RATE_FLOOR)


def _discount_factor(curve: pd.Series, tenor_yr: float) -> float:
    """
    Annual-compounding spot discount factor: D(t) = 1 / (1 + r(t))^t.
    Returns 1.0 for tenor ≤ 0.
    """
    if tenor_yr <= 0.0:
        return 1.0
    r = _interpolate_rate(curve, tenor_yr)
    return 1.0 / (1.0 + r) ** tenor_yr


def load_base_curve() -> pd.Series:
    """
    Fetch the current Treasury yield curve from FRED and return it as a
    decimal ``pd.Series`` (tenor in years → rate as decimal).

    Falls back to the hardcoded curve in ``data.fred_loader`` if the API
    is unavailable.

    Returns
    -------
    pd.Series
        Index: tenor_yr (float), values: rate (decimal).
    """
    from data.fred_loader import fetch_yield_curve   # local import — avoids circular dep

    df = fetch_yield_curve()
    curve = pd.Series(
        df["rate_pct"].values / 100.0,
        index=df["tenor_yr"].values,
        name="rate_decimal",
    )
    return curve.sort_index()


# ---------------------------------------------------------------------------
# Cash flow generation
# ---------------------------------------------------------------------------

def _build_cashflows(instr: Instrument) -> list[tuple[float, float]]:
    """
    Build an annual-coupon cash flow schedule for a single instrument.

    Fixed-rate
        Annual coupon payments from year 1 to maturity, plus principal
        at maturity.  Fractional final periods receive a prorated coupon.

    Floating-rate (repricing at par)
        Cash flows are generated **only up to the next repricing event**
        (= ``repricing_tenor_months / 12`` years, capped at effective
        maturity).  At that point the full principal is returned at par,
        reflecting the assumption that the instrument refinances at market.

    Non-maturity deposits without a contractual maturity
        Effective maturity is taken from ``behavioural_maturity_years``
        (if set), otherwise from ``repricing_tenor_months / 12``.

    Parameters
    ----------
    instr :
        The instrument to schedule.

    Returns
    -------
    list of (time_yr: float, cash_flow: float)
        Sorted ascending.  All values in $M.
    """
    N = instr.notional
    c = instr.coupon

    if instr.rate_type == "fixed":
        # Use effective maturity (behavioural if set, else contractual)
        T = instr.effective_maturity_years
        if T is None:
            # Fixed non-maturity instrument — use repricing tenor as proxy
            T = max(instr.repricing_tenor_months / 12.0, 0.5)
    else:
        # Floating: model only the period until the next repricing event
        T = instr.repricing_tenor_months / 12.0
        eff_mat = instr.effective_maturity_years
        if eff_mat is not None:
            T = min(T, eff_mat)
        # Safety: if T is somehow zero, treat as 1-day maturity
        T = max(T, 1.0 / 365.0)

    T = max(T, 1.0 / 365.0)  # universal safety floor

    cashflows: list[tuple[float, float]] = []
    n_full = int(T)  # number of complete annual coupon periods before maturity

    for k in range(1, n_full + 1):
        t = float(k)
        if t < T - 1e-9:
            cashflows.append((t, N * c))           # interior coupon
        else:
            cashflows.append((t, N * (1.0 + c)))  # final coupon + principal

    # Fractional tail period
    frac = T - n_full
    if frac > 1e-6:
        cashflows.append((T, N * c * frac + N))   # prorated coupon + principal

    return sorted(cashflows, key=lambda x: x[0])


def _pv_and_macaulay(
    cashflows: list[tuple[float, float]],
    curve: pd.Series,
) -> tuple[float, float]:
    """
    Discount a cash flow stream and return (PV, Macaulay duration).

    Macaulay duration = Σ(t × PV(CF_t)) / PV_total.
    Returns (pv, 0.0) when PV ≈ 0 to avoid division by zero.
    """
    pv = 0.0
    weighted_t = 0.0

    for t, cf in cashflows:
        df  = _discount_factor(curve, t)
        pv_cf = cf * df
        pv        += pv_cf
        weighted_t += t * pv_cf

    if abs(pv) < 1e-9:
        return pv, 0.0

    return pv, weighted_t / pv


# ---------------------------------------------------------------------------
# Public utility: modified duration (keeps original placeholder signature)
# ---------------------------------------------------------------------------

def modified_duration(
    cash_flows: np.ndarray,
    times_yr:   np.ndarray,
    discount_rate: float,
) -> float:
    """
    Compute modified duration for a cash flow stream at a flat discount rate.

    Parameters
    ----------
    cash_flows :
        Array of cash flow amounts.
    times_yr :
        Array of cash flow timings in years (same length as ``cash_flows``).
    discount_rate :
        Flat annual discount rate (decimal).

    Returns
    -------
    float
        Modified duration in years.

    Notes
    -----
    Modified duration = Macaulay duration / (1 + y), where y is the flat
    yield.  Annual compounding is assumed throughout.
    """
    cash_flows    = np.asarray(cash_flows, dtype=float)
    times_yr      = np.asarray(times_yr,   dtype=float)
    discount_rate = max(float(discount_rate), _RATE_FLOOR)

    if len(cash_flows) == 0 or len(cash_flows) != len(times_yr):
        return 0.0

    dfs = 1.0 / (1.0 + discount_rate) ** times_yr
    pv  = (cash_flows * dfs).sum()

    if abs(pv) < 1e-9:
        return 0.0

    macaulay = (times_yr * cash_flows * dfs).sum() / pv
    return macaulay / (1.0 + discount_rate)


# ---------------------------------------------------------------------------
# Core private valuation engine
# ---------------------------------------------------------------------------

def _value_all_instruments(
    balance_sheet: BalanceSheet,
    curve: pd.Series,
) -> tuple[float, float, list[dict]]:
    """
    Value every instrument against a given yield curve.

    Returns
    -------
    (pv_assets, pv_liabilities, rows)
        ``rows`` is a list of per-instrument dicts used to build DataFrames.
        PV and duration figures are computed under ``curve``.
    """
    pv_assets      = 0.0
    pv_liabilities = 0.0
    rows: list[dict] = []

    for instr in balance_sheet.assets + balance_sheet.liabilities:
        cfs             = _build_cashflows(instr)
        pv, mac_dur     = _pv_and_macaulay(cfs, curve)

        # Representative tenor for YTM look-up (used in modified duration)
        T_rep = instr.effective_maturity_years
        if T_rep is None:
            T_rep = instr.repricing_tenor_months / 12.0
        T_rep = max(T_rep, 1.0 / 365.0)

        ytm     = _interpolate_rate(curve, T_rep)
        mod_dur = mac_dur / (1.0 + ytm)

        # DV01: approximate $ change in PV for a +1bp upward shift
        dv01 = -pv * mod_dur * 0.0001   # negative: PV falls when rates rise

        if instr.side == "asset":
            pv_assets += pv
        else:
            pv_liabilities += pv

        rows.append({
            "side":                   instr.side,
            "name":                   instr.name,
            "notional":               instr.notional,
            "rate_type":              instr.rate_type,
            "coupon_pct":             round(instr.coupon * 100, 3),
            "repricing_tenor_months": instr.repricing_tenor_months,
            "effective_maturity_yr":  round(T_rep, 3),
            "pv":                     pv,
            "macaulay_duration":      round(mac_dur, 4),
            "modified_duration":      round(mod_dur, 4),
            "dv01":                   round(dv01, 5),
        })

    return pv_assets, pv_liabilities, rows


def _portfolio_durations(
    rows: list[dict],
    pv_assets: float,
    pv_liabilities: float,
) -> tuple[float, float, float]:
    """
    Compute PV-weighted average modified duration for assets and liabilities,
    and the duration gap.

    Duration Gap = DA − (PV_L / PV_A) × DL
    """
    asset_rows = [r for r in rows if r["side"] == "asset"]
    liab_rows  = [r for r in rows if r["side"] == "liability"]

    def _weighted_dur(item_rows: list[dict], total_pv: float) -> float:
        if total_pv < 1e-9:
            return 0.0
        return sum(r["pv"] * r["modified_duration"] for r in item_rows) / total_pv

    da = _weighted_dur(asset_rows, pv_assets)
    dl = _weighted_dur(liab_rows, pv_liabilities)

    if pv_assets > 1e-9:
        dg = da - (pv_liabilities / pv_assets) * dl
    else:
        dg = 0.0

    return da, dl, dg


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EveResult:
    """
    Complete EVE sensitivity result for one rate scenario.

    Attributes
    ----------
    scenario_name :
        Machine-readable key (e.g. ``"parallel_up_100"``).
    scenario_label :
        Human-readable description (e.g. ``"+100 bp Parallel"``).
    pv_assets :
        Present value of all assets under the shocked curve ($M).
    pv_liabilities :
        Present value of all liabilities under the shocked curve ($M).
    eve :
        EVE = pv_assets − pv_liabilities under the shocked curve ($M).
    base_eve :
        EVE under the base (unshocked) curve ($M).
    delta_eve :
        eve − base_eve ($M).
    duration_assets :
        PV-weighted average modified duration of all assets (years).
    duration_liabilities :
        PV-weighted average modified duration of all liabilities (years).
    duration_gap :
        DA − (PV_L / PV_A) × DL (years).
    bp01 :
        Central-difference EVE sensitivity to +1 bp on the base curve ($M).
        Positive = EVE rises when rates increase (rare; typical bank has negative BP01).
    by_instrument :
        Per-instrument breakdown DataFrame (see column list below).

    ``by_instrument`` columns
    -------------------------
    side, name, notional, rate_type, coupon_pct,
    repricing_tenor_months, effective_maturity_yr,
    pv_base, pv_shocked, delta_pv, delta_pv_pct,
    macaulay_duration, modified_duration, dv01
    """

    scenario_name:        str
    scenario_label:       str
    pv_assets:            float
    pv_liabilities:       float
    eve:                  float
    base_eve:             float
    delta_eve:            float
    duration_assets:      float
    duration_liabilities: float
    duration_gap:         float
    bp01:                 float
    by_instrument:        pd.DataFrame = field(repr=False)

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def pct_change_eve(self) -> float:
        """Change in EVE as % of base EVE.  Zero for the base scenario."""
        if abs(self.base_eve) < 1e-9:
            return 0.0
        return self.delta_eve / abs(self.base_eve) * 100

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def display(self) -> None:
        """Pretty-print the full EVE sensitivity report to stdout."""
        SEP  = "=" * 100
        DASH = "-" * 100

        print(SEP)
        print(
            f"  EVE SENSITIVITY  |  Scenario: {self.scenario_label}  "
            f"|  [{self.scenario_name}]"
        )
        print(DASH)
        print(f"  {'':55s}  {'Base':>10s}   {'Shocked':>10s}   {'Delta':>10s}")
        print(DASH)
        print(
            f"  {'PV of Assets':55s}"
            f"  {self.base_eve + self.pv_liabilities - self.pv_liabilities:>10.2f}"
            f"   {self.pv_assets:>10.2f}   {'':>10s}"
        )

        # Show asset and liability PVs side by side
        base_pv_a = self.by_instrument.query("side=='asset'")["pv_base"].sum()
        base_pv_l = self.by_instrument.query("side=='liability'")["pv_base"].sum()

        print(
            f"  {'PV of Assets ($M)':55s}"
            f"  {base_pv_a:>10.2f}   {self.pv_assets:>10.2f}"
            f"   {self.pv_assets - base_pv_a:>+10.2f}"
        )
        print(
            f"  {'PV of Liabilities ($M)':55s}"
            f"  {base_pv_l:>10.2f}   {self.pv_liabilities:>10.2f}"
            f"   {self.pv_liabilities - base_pv_l:>+10.2f}"
        )
        print(DASH)
        print(
            f"  {'EVE ($M)':55s}"
            f"  {self.base_eve:>10.2f}   {self.eve:>10.2f}"
            f"   {self.delta_eve:>+10.2f}"
        )
        print(
            f"  {'EVE change (%)':55s}"
            f"  {'':>10s}   {'':>10s}   {self.pct_change_eve:>+10.2f}%"
        )

        print(f"\n  DURATION METRICS (shocked curve)")
        print(DASH)
        print(f"  {'Duration of Assets':55s}  {self.duration_assets:>10.3f}  years")
        print(f"  {'Duration of Liabilities':55s}  {self.duration_liabilities:>10.3f}  years")
        print(f"  {'Duration Gap  (DA − L/A × DL)':55s}  {self.duration_gap:>+10.3f}  years")
        print(f"  {'BP01  (EVE Δ per +1bp, $M)':55s}  {self.bp01:>+10.4f}  $M/bp")

        print(f"\n  PER-INSTRUMENT BREAKDOWN")
        print(DASH)
        cols = [
            "side", "name", "notional", "rate_type", "coupon_pct",
            "effective_maturity_yr", "pv_base", "pv_shocked",
            "delta_pv", "delta_pv_pct", "modified_duration", "dv01",
        ]
        avail = [c for c in cols if c in self.by_instrument.columns]
        with pd.option_context(
            "display.float_format", "{:,.3f}".format,
            "display.max_colwidth", 28,
        ):
            print(self.by_instrument[avail].to_string(index=False))
        print(SEP)


# ---------------------------------------------------------------------------
# Main EVE computation
# ---------------------------------------------------------------------------

def compute_eve(
    balance_sheet: BalanceSheet,
    scenario: RateScenario,
) -> EveResult:
    """
    Compute EVE (Economic Value of Equity) under a rate scenario.

    For each instrument the cash flow schedule is discounted using spot
    rates interpolated from the shocked yield curve.  Fixed-rate instruments
    change in value because the same cash flows are discounted at higher/lower
    rates.  Floating-rate instruments change in value only over the short
    period to their next repricing event (repricing-at-par assumption).

    EVE = PV(Assets) − PV(Liabilities)

    Parameters
    ----------
    balance_sheet :
        Populated balance sheet instance.
    scenario :
        ``RateScenario`` carrying both ``base_curve`` and ``shocked_curve``
        (tenor-indexed ``pd.Series`` with rates in **decimal**).

    Returns
    -------
    EveResult
        Structured result with EVE, Δ EVE, duration metrics, BP01, and a
        per-instrument breakdown DataFrame.
    """
    base_curve    = scenario.base_curve
    shocked_curve = scenario.shocked_curve

    # --- Value under base curve ---
    pv_a_base, pv_l_base, base_rows = _value_all_instruments(balance_sheet, base_curve)
    eve_base = pv_a_base - pv_l_base
    base_pv_map = {r["name"]: r["pv"] for r in base_rows}

    # --- Value under shocked curve ---
    pv_a_shock, pv_l_shock, shock_rows = _value_all_instruments(balance_sheet, shocked_curve)
    eve_shock  = pv_a_shock - pv_l_shock
    delta_eve  = eve_shock - eve_base

    # --- Duration metrics (from shocked-curve valuation) ---
    da, dl, dg = _portfolio_durations(shock_rows, pv_a_shock, pv_l_shock)

    # --- BP01: central difference on the base curve ---
    bp01_curve_up   = (base_curve + 0.0001).clip(lower=_RATE_FLOOR)
    bp01_curve_down = (base_curve - 0.0001).clip(lower=_RATE_FLOOR)
    pv_a_up, pv_l_up, _ = _value_all_instruments(balance_sheet, bp01_curve_up)
    pv_a_dn, pv_l_dn, _ = _value_all_instruments(balance_sheet, bp01_curve_down)
    eve_up  = pv_a_up - pv_l_up
    eve_dn  = pv_a_dn - pv_l_dn
    bp01    = (eve_up - eve_dn) / 2.0

    # --- Build per-instrument DataFrame ---
    rows_out: list[dict] = []
    for r in shock_rows:
        name         = r["name"]
        pv_base_instr    = base_pv_map.get(name, r["pv"])
        pv_shocked_instr = r["pv"]
        dpv              = pv_shocked_instr - pv_base_instr
        dpv_pct          = (dpv / max(abs(pv_base_instr), 1e-9)) * 100

        rows_out.append({
            "side":                   r["side"],
            "name":                   r["name"],
            "notional":               r["notional"],
            "rate_type":              r["rate_type"],
            "coupon_pct":             r["coupon_pct"],
            "repricing_tenor_months": r["repricing_tenor_months"],
            "effective_maturity_yr":  r["effective_maturity_yr"],
            "pv_base":                round(pv_base_instr,    3),
            "pv_shocked":             round(pv_shocked_instr, 3),
            "delta_pv":               round(dpv,              3),
            "delta_pv_pct":           round(dpv_pct,          2),
            "macaulay_duration":      r["macaulay_duration"],
            "modified_duration":      r["modified_duration"],
            "dv01":                   r["dv01"],
        })

    by_instrument = pd.DataFrame(rows_out)

    logger.info(
        "EVE | scenario=%s  base=%.2f  shocked=%.2f  delta=%+.2f  "
        "dur_gap=%.3f  bp01=%+.4f",
        scenario.name, eve_base, eve_shock, delta_eve, dg, bp01,
    )

    return EveResult(
        scenario_name=scenario.name,
        scenario_label=scenario.label,
        pv_assets=round(pv_a_shock, 4),
        pv_liabilities=round(pv_l_shock, 4),
        eve=round(eve_shock, 4),
        base_eve=round(eve_base, 4),
        delta_eve=round(delta_eve, 4),
        duration_assets=round(da, 4),
        duration_liabilities=round(dl, 4),
        duration_gap=round(dg, 4),
        bp01=round(bp01, 5),
        by_instrument=by_instrument,
    )


# ---------------------------------------------------------------------------
# Scenario comparison table
# ---------------------------------------------------------------------------

def eve_sensitivity_table(
    balance_sheet: BalanceSheet,
    scenarios: dict[str, RateScenario],
) -> pd.DataFrame:
    """
    Run EVE for every scenario and return a clean comparison DataFrame.

    Parameters
    ----------
    balance_sheet :
        Populated balance sheet instance.
    scenarios :
        Mapping of scenario name → ``RateScenario``.  Should include a
        ``"base"`` entry (``shocked_curve == base_curve``) so that
        ``delta_eve`` is meaningful across rows.

    Returns
    -------
    pd.DataFrame
        Columns: ``scenario``, ``label``, ``pv_assets``, ``pv_liabilities``,
        ``eve``, ``delta_eve``, ``pct_change_eve``, ``duration_assets``,
        ``duration_liabilities``, ``duration_gap``, ``bp01``.
        Suitable for direct use in a Streamlit ``st.dataframe()`` call.
    """
    rows: list[dict] = []

    for name, scenario in scenarios.items():
        r = compute_eve(balance_sheet, scenario)
        rows.append({
            "scenario":           name,
            "label":              scenario.label,
            "pv_assets":          round(r.pv_assets,            2),
            "pv_liabilities":     round(r.pv_liabilities,       2),
            "eve":                round(r.eve,                  2),
            "delta_eve":          round(r.delta_eve,            2),
            "pct_change_eve":     round(r.pct_change_eve,       2),
            "duration_assets":    round(r.duration_assets,      3),
            "duration_liabilities": round(r.duration_liabilities, 3),
            "duration_gap":       round(r.duration_gap,         3),
            "bp01":               round(r.bp01,                 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NII functions (using RateScenario yield curves)
# ---------------------------------------------------------------------------

def project_nii(
    balance_sheet: BalanceSheet,
    scenario: RateScenario,
    horizon_yr: float = 1.0,
) -> float:
    """
    Project Net Interest Income over a horizon under a rate scenario.

    Uses the full shocked yield curve to compute the rate change at each
    instrument's specific repricing tenor, giving correct results for
    non-parallel shock scenarios (twists, flatteners, etc.).

    Parameters
    ----------
    balance_sheet :
        Populated balance sheet instance.
    scenario :
        ``RateScenario`` with ``base_curve`` and ``shocked_curve``.
    horizon_yr :
        NII projection horizon in years (default 1 year).

    Returns
    -------
    float
        Projected NII in $M (base NII + ΔNII from the shock).
    """
    # Import here to avoid circular dependency at module level
    from models.repricing import _repricing_events  # package-internal use

    base_curve    = scenario.base_curve
    shocked_curve = scenario.shocked_curve
    base_nii      = balance_sheet.net_interest_income
    horizon_months = int(round(horizon_yr * 12))

    delta_nii = 0.0
    for instr in balance_sheet.assets + balance_sheet.liabilities:
        events = _repricing_events(instr, horizon_months)
        if not events:
            continue

        # Use the tenor of the first repricing event to look up the shock
        t_first_yr = events[0] / 12.0

        r_shocked = _interpolate_rate(shocked_curve, t_first_yr)
        r_base    = _interpolate_rate(base_curve,    t_first_yr)
        shock_at_tenor = r_shocked - r_base          # decimal (e.g. 0.01 = 100bp)

        # Fraction of the horizon remaining after first repricing
        t_capped      = min(t_first_yr, horizon_yr)
        income_frac   = (horizon_yr - t_capped) / horizon_yr

        sign = 1.0 if instr.side == "asset" else -1.0
        delta_nii += sign * instr.notional * shock_at_tenor * income_frac

    return base_nii + delta_nii


def nii_sensitivity_table(
    balance_sheet: BalanceSheet,
    scenarios: dict[str, RateScenario],
    horizon_yr: float = 1.0,
) -> pd.DataFrame:
    """
    Build a scenario comparison table for NII sensitivity.

    Parameters
    ----------
    balance_sheet :
        Populated balance sheet instance.
    scenarios :
        Mapping of scenario name → ``RateScenario``.
    horizon_yr :
        NII projection horizon in years (default 1 year).

    Returns
    -------
    pd.DataFrame
        Columns: ``scenario``, ``label``, ``nii``, ``delta_nii``,
        ``pct_change``.  Suitable for Streamlit display.
    """
    base_nii = balance_sheet.net_interest_income
    rows: list[dict] = []

    for name, scenario in scenarios.items():
        nii = project_nii(balance_sheet, scenario, horizon_yr)
        delta = nii - base_nii
        pct   = (delta / abs(base_nii) * 100) if abs(base_nii) > 1e-9 else 0.0
        rows.append({
            "scenario":   name,
            "label":      scenario.label,
            "nii":        round(nii,   3),
            "delta_nii":  round(delta, 3),
            "pct_change": round(pct,   2),
        })

    return pd.DataFrame(rows)
