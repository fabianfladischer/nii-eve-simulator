"""
scenarios/rate_scenarios.py
---------------------------
Interest rate scenario construction for the NII/EVE simulator.

Responsibilities:
- Define the RateScenario dataclass
- Build shocked yield curves from a base curve and a scenario specification
- Support parallel shifts, twists, bear/bull flatteners and steepeners
- Provide convenience factory functions aligned with config.RATE_SCENARIOS
- Visualise all scenarios on a single Plotly chart

BCBS/EBA IRRBB non-parallel shock calibration
----------------------------------------------
The six non-parallel scenarios follow the standard anchor-based profile:
  - Short end plateau: full shock applied at tenors ≤ 2Y
  - Linear transition: shock tapers between 2Y and 10Y
  - Long end plateau:  residual shock held flat at tenors ≥ 10Y

Scenario    | Short (≤2Y) | Long (≥10Y)
------------|-------------|------------
Bear Steep  |   +200 bp   |   +100 bp
Bull Flat   |   −100 bp   |   −200 bp
Bear Flat   |   +200 bp   |      0 bp
Bull Steep  |      0 bp   |   −200 bp
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RATE_FLOOR: float = 0.0001  # 1 bp minimum; prevents negative rates in downside shocks

# BCBS/EBA anchor profiles for non-parallel scenarios.
# Each tuple: (list_of_anchor_tenors_yr, list_of_shock_bp_at_each_anchor)
# np.interp uses linear interpolation between anchors and flat extrapolation
# beyond the first / last anchor.
_BEAR_STEEPENER_ANCHORS = ([0.0, 2.0, 10.0, 30.0], [200,  200,  100,  100])
_BULL_FLATTENER_ANCHORS  = ([0.0, 2.0, 10.0, 30.0], [-100, -100, -200, -200])
_BEAR_FLATTENER_ANCHORS  = ([0.0, 2.0, 10.0, 30.0], [200,  200,    0,    0])
_BULL_STEEPENER_ANCHORS  = ([0.0, 2.0, 10.0, 30.0], [0,      0, -200, -200])


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RateScenario:
    """
    A fully-specified interest rate scenario.

    Attributes
    ----------
    name:
        Machine-readable identifier matching a key in ``config.RATE_SCENARIOS``.
    label:
        Human-readable description for display in the UI.
    base_curve:
        Yield curve prior to shock.  Index: tenor in years; values: rates (decimal).
    shocked_curve:
        Yield curve after shock.  Same structure as ``base_curve``.
    """

    name: str
    label: str
    base_curve: pd.Series = field(default_factory=pd.Series)
    shocked_curve: pd.Series = field(default_factory=pd.Series)

    @property
    def shock_curve(self) -> pd.Series:
        """Point-wise shock magnitude (shocked − base) in decimal form."""
        if not self.base_curve.index.equals(self.shocked_curve.index):
            # Re-align on the union of both indices via index-aware interpolation
            all_tenors = self.base_curve.index.union(self.shocked_curve.index)
            base = self.base_curve.reindex(all_tenors).interpolate(method="index")
            shocked = self.shocked_curve.reindex(all_tenors).interpolate(method="index")
            return shocked - base
        return self.shocked_curve - self.base_curve

    @property
    def shock_curve_bp(self) -> pd.Series:
        """Point-wise shock magnitude in basis points."""
        return self.shock_curve * 10_000.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_profile(
    base_curve: pd.Series,
    anchor_tenors: list[float],
    anchor_shocks_bp: list[float],
    floor: bool = True,
) -> pd.Series:
    """
    Apply a tenor-varying shock profile to a base yield curve.

    The shock at each tenor is determined by linear interpolation of the
    anchor points; beyond the anchor range the nearest anchor value is used
    (numpy's flat-extrapolation default).

    Parameters
    ----------
    base_curve:
        Tenor-indexed yield curve (rates in decimal).
    anchor_tenors:
        Tenor points (years) at which the shock magnitude is defined.
        Must be in ascending order.
    anchor_shocks_bp:
        Shock magnitude (basis points) at each anchor tenor.
    floor:
        When True, shocked rates are clipped at ``_RATE_FLOOR``.

    Returns
    -------
    pd.Series
        Shocked yield curve, same index as ``base_curve``, same name.
    """
    tenors = base_curve.index.to_numpy(dtype=float)
    shock_bp = np.interp(tenors, anchor_tenors, anchor_shocks_bp)
    shocked = base_curve.to_numpy(dtype=float) + shock_bp / 10_000.0
    if floor:
        shocked = np.maximum(shocked, _RATE_FLOOR)
    return pd.Series(shocked, index=base_curve.index, name=base_curve.name)


# ---------------------------------------------------------------------------
# Public shock constructors
# ---------------------------------------------------------------------------


def apply_parallel_shift(base_curve: pd.Series, shock_bp: float) -> pd.Series:
    """
    Apply a uniform parallel shift to every tenor of the base curve.

    Parameters
    ----------
    base_curve:
        Tenor-indexed yield curve (rates in decimal).
    shock_bp:
        Shock magnitude in basis points (positive = rates up, negative = rates down).

    Returns
    -------
    pd.Series
        Shocked yield curve (rates in decimal).
    """
    shocked = base_curve.to_numpy(dtype=float) + shock_bp / 10_000.0
    shocked = np.maximum(shocked, _RATE_FLOOR)
    return pd.Series(shocked, index=base_curve.index, name=base_curve.name)


def apply_bear_steepener(base_curve: pd.Series) -> pd.Series:
    """
    Bear Steepener: short end +200 bp, long end +100 bp.

    BCBS/EBA IRRBB profile:
    - ≤ 2Y : +200 bp (plateau)
    - 2Y–10Y: linear taper from +200 bp to +100 bp
    - ≥ 10Y: +100 bp (plateau)

    Parameters
    ----------
    base_curve:
        Tenor-indexed yield curve (rates in decimal).

    Returns
    -------
    pd.Series
        Shocked yield curve (rates in decimal).
    """
    return _apply_profile(base_curve, *_BEAR_STEEPENER_ANCHORS)


def apply_bull_flattener(base_curve: pd.Series) -> pd.Series:
    """
    Bull Flattener: short end −100 bp, long end −200 bp.

    BCBS/EBA IRRBB profile:
    - ≤ 2Y : −100 bp (plateau)
    - 2Y–10Y: linear taper from −100 bp to −200 bp
    - ≥ 10Y: −200 bp (plateau)

    Parameters
    ----------
    base_curve:
        Tenor-indexed yield curve (rates in decimal).

    Returns
    -------
    pd.Series
        Shocked yield curve (rates in decimal).
    """
    return _apply_profile(base_curve, *_BULL_FLATTENER_ANCHORS)


def apply_bear_flattener(base_curve: pd.Series) -> pd.Series:
    """
    Bear Flattener: short end +200 bp, long end flat (0 bp).

    BCBS/EBA IRRBB profile:
    - ≤ 2Y : +200 bp (plateau)
    - 2Y–10Y: linear taper from +200 bp to 0 bp
    - ≥ 10Y: 0 bp (no change)

    Parameters
    ----------
    base_curve:
        Tenor-indexed yield curve (rates in decimal).

    Returns
    -------
    pd.Series
        Shocked yield curve (rates in decimal).
    """
    return _apply_profile(base_curve, *_BEAR_FLATTENER_ANCHORS)


def apply_bull_steepener(base_curve: pd.Series) -> pd.Series:
    """
    Bull Steepener: short end flat (0 bp), long end −200 bp.

    BCBS/EBA IRRBB profile:
    - ≤ 2Y : 0 bp (no change)
    - 2Y–10Y: linear taper from 0 bp to −200 bp
    - ≥ 10Y: −200 bp (plateau)

    Parameters
    ----------
    base_curve:
        Tenor-indexed yield curve (rates in decimal).

    Returns
    -------
    pd.Series
        Shocked yield curve (rates in decimal).
    """
    return _apply_profile(base_curve, *_BULL_STEEPENER_ANCHORS)


def apply_twist(
    base_curve: pd.Series,
    short_shock_bp: float,
    long_shock_bp: float,
    pivot_tenor_yr: float,
) -> pd.Series:
    """
    Apply a twist: shock interpolates linearly from short to long ends,
    passing through zero at the pivot tenor.

    Parameters
    ----------
    base_curve:
        Tenor-indexed yield curve (rates in decimal).
    short_shock_bp:
        Shock at the short end (tenor → 0).
    long_shock_bp:
        Shock at the long end (tenor → ∞).
    pivot_tenor_yr:
        Tenor (years) at which the net shock is zero.

    Returns
    -------
    pd.Series
        Shocked yield curve (rates in decimal).
    """
    max_tenor = float(base_curve.index.max())
    end_tenor = max(max_tenor, pivot_tenor_yr + 1.0)
    anchors_yr = [0.0, pivot_tenor_yr, end_tenor]
    anchors_bp = [short_shock_bp, 0.0, long_shock_bp]
    return _apply_profile(base_curve, anchors_yr, anchors_bp)


def apply_custom_scenario(
    base_curve: pd.Series,
    tenor_shocks_bp: dict[float, float],
) -> pd.Series:
    """
    Apply a user-defined shock profile at arbitrary tenors, interpolated across the curve.

    The shock at tenors not explicitly listed is linearly interpolated from the
    nearest neighbours.  Outside the supplied tenor range, the nearest boundary
    shock is held flat (no extrapolation).

    Parameters
    ----------
    base_curve:
        Tenor-indexed yield curve (rates in decimal).
    tenor_shocks_bp:
        ``{tenor_yr: shock_bp}`` mapping.  Must contain at least one entry.
        Shocks at other tenors are linearly interpolated between the supplied points.

    Returns
    -------
    pd.Series
        Shocked yield curve (rates in decimal).

    Examples
    --------
    >>> shocked = apply_custom_scenario(base_curve, {0.25: 50, 2.0: 100, 10.0: 75})
    """
    if not tenor_shocks_bp:
        raise ValueError("tenor_shocks_bp must not be empty.")
    anchors = sorted(tenor_shocks_bp.items())  # ascending tenor order
    anchor_tenors = [a[0] for a in anchors]
    anchor_bps    = [a[1] for a in anchors]
    return _apply_profile(base_curve, anchor_tenors, anchor_bps)


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------


def build_scenarios(base_curve: pd.Series) -> dict[str, RateScenario]:
    """
    Construct all scenarios defined in ``config.RATE_SCENARIOS`` from a base curve.

    A base (no-shock) scenario is always included under the key ``"base"``.

    Parameters
    ----------
    base_curve:
        Current market yield curve (tenor-indexed, rates in decimal).

    Returns
    -------
    dict[str, RateScenario]
        Ordered mapping of scenario name → ``RateScenario``.
        The base scenario is always first; remaining scenarios follow the
        order defined in ``config.RATE_SCENARIOS``.

    Raises
    ------
    ValueError
        If ``config.RATE_SCENARIOS`` contains an unrecognised ``type`` field.
    """
    scenarios: dict[str, RateScenario] = {}

    # Always include the no-shock base
    scenarios["base"] = RateScenario(
        name="base",
        label="Base (No Shock)",
        base_curve=base_curve,
        shocked_curve=base_curve.copy(),
    )

    dispatch = {
        "parallel":        _build_parallel,
        "bear_steepener":  _build_bear_steepener,
        "bull_flattener":  _build_bull_flattener,
        "bear_flattener":  _build_bear_flattener,
        "bull_steepener":  _build_bull_steepener,
        "twist":           _build_twist,
        "custom":          _build_custom,
    }

    for name, spec in config.RATE_SCENARIOS.items():
        stype = spec["type"]
        builder = dispatch.get(stype)
        if builder is None:
            logger.warning(
                "Unknown scenario type %r for scenario %r — skipping.", stype, name
            )
            continue

        shocked = builder(base_curve, spec)
        scenarios[name] = RateScenario(
            name=name,
            label=spec["label"],
            base_curve=base_curve,
            shocked_curve=shocked,
        )
        logger.debug("Built scenario %r: %s", name, spec["label"])

    logger.info("Built %d scenarios (including base).", len(scenarios))
    return scenarios


# ---------------------------------------------------------------------------
# Private builder helpers (one per scenario type)
# ---------------------------------------------------------------------------


def _build_parallel(base_curve: pd.Series, spec: dict) -> pd.Series:
    return apply_parallel_shift(base_curve, spec["shock_bp"])


def _build_bear_steepener(base_curve: pd.Series, spec: dict) -> pd.Series:  # noqa: ARG001
    return apply_bear_steepener(base_curve)


def _build_bull_flattener(base_curve: pd.Series, spec: dict) -> pd.Series:  # noqa: ARG001
    return apply_bull_flattener(base_curve)


def _build_bear_flattener(base_curve: pd.Series, spec: dict) -> pd.Series:  # noqa: ARG001
    return apply_bear_flattener(base_curve)


def _build_bull_steepener(base_curve: pd.Series, spec: dict) -> pd.Series:  # noqa: ARG001
    return apply_bull_steepener(base_curve)


def _build_twist(base_curve: pd.Series, spec: dict) -> pd.Series:
    return apply_twist(
        base_curve,
        short_shock_bp=spec["short_shock_bp"],
        long_shock_bp=spec["long_shock_bp"],
        pivot_tenor_yr=spec["pivot_tenor_yr"],
    )


def _build_custom(base_curve: pd.Series, spec: dict) -> pd.Series:
    return apply_custom_scenario(base_curve, spec["tenor_shocks_bp"])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_scenarios(
    scenarios: dict[str, RateScenario] | None = None,
    base_curve: pd.Series | None = None,
    title: str = "Interest Rate Shock Scenarios",
    show: bool = True,
) -> go.Figure:
    """
    Plot multiple shocked yield curves on a single chart.

    Parameters
    ----------
    scenarios:
        Output of :func:`build_scenarios`.  When None, ``base_curve`` must be
        provided and all configured scenarios are built automatically.
    base_curve:
        Base yield curve (tenor-indexed, decimal).  Used only when *scenarios* is None.
    title:
        Chart title.
    show:
        Call ``fig.show()`` before returning when True (default).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if scenarios is None:
        if base_curve is None:
            from models.risk_metrics import load_base_curve
            base_curve = load_base_curve()
        scenarios = build_scenarios(base_curve)

    colours = [
        "#1f77b4",  # base  – blue
        "#d62728",  # red
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # grey
        "#17becf",  # teal
    ]

    fig = go.Figure()

    for i, (name, sc) in enumerate(scenarios.items()):
        curve_pct = sc.shocked_curve * 100.0  # decimal → percent for display
        is_base = name == "base"
        fig.add_trace(go.Scatter(
            x=curve_pct.index.tolist(),
            y=curve_pct.values.tolist(),
            mode="lines+markers",
            name=sc.label,
            line=dict(
                color=colours[i % len(colours)],
                width=2.5 if is_base else 1.8,
                dash="solid" if is_base else "dot",
            ),
            marker=dict(size=5 if is_base else 4),
            hovertemplate=(
                f"<b>{sc.label}</b><br>"
                "Tenor: %{x:.2f} yr<br>"
                "Yield: %{y:.3f}%<extra></extra>"
            ),
        ))

    tenor_labels = {
        1 / 12: "1M", 3 / 12: "3M", 6 / 12: "6M",
        1.0: "1Y", 2.0: "2Y", 5.0: "5Y",
        7.0: "7Y", 10.0: "10Y", 20.0: "20Y", 30.0: "30Y",
    }

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(
            title="Tenor (years)",
            tickvals=sorted(tenor_labels.keys()),
            ticktext=list(tenor_labels.values()),
            showgrid=True,
            gridcolor="#e5e5e5",
        ),
        yaxis=dict(
            title="Yield (%)",
            showgrid=True,
            gridcolor="#e5e5e5",
        ),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(x=1.01, y=1.0, xanchor="left", borderwidth=1),
        margin=dict(t=60, b=60, l=60, r=220),
    )

    if show:
        fig.show()

    return fig
