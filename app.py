"""
app.py
------
Streamlit dashboard for the NII / EVE Interest Rate Risk Simulator.

Pages (sidebar navigation)
---------------------------
1. Yield Curve  — Base vs shocked curve overlay, rate table
2. Balance Sheet — Instruments, repricing gap chart
3. NII Impact   — Delta NII by bucket, instant vs ramp, all-scenario sensitivity
4. EVE Impact   — Delta EVE by scenario, duration analysis, per-instrument breakdown
5. Risk Summary — Heatmap, key risk metrics, regulatory commentary

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Design constants
# ---------------------------------------------------------------------------

_C = {
    "navy":        "#0D2B55",
    "blue":        "#1565C0",
    "light_blue":  "#90CAF9",
    "red":         "#C62828",
    "light_red":   "#FFCDD2",
    "green":       "#2E7D32",
    "light_green": "#C8E6C9",
    "amber":       "#E65100",
    "gray":        "#546E7A",
    "grid":        "#ECEFF1",
    "bg":          "#FFFFFF",
}

_TENOR_LABELS: dict[float, str] = {
    1 / 12: "1M", 3 / 12: "3M", 6 / 12: "6M",
    1.0: "1Y", 2.0: "2Y", 5.0: "5Y",
    7.0: "7Y", 10.0: "10Y", 20.0: "20Y", 30.0: "30Y",
}

_SLIDER_TENORS: list[tuple[float, str]] = [
    (1 / 12, "1M"), (3 / 12, "3M"), (6 / 12, "6M"),
    (1.0, "1Y"), (2.0, "2Y"), (5.0, "5Y"),
    (7.0, "7Y"), (10.0, "10Y"), (20.0, "20Y"), (30.0, "30Y"),
]


# ---------------------------------------------------------------------------
# Data loading — cached
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86_400, show_spinner=False)
def _load_curve_data() -> dict:
    """Fetch FRED curve; return curve Series + metadata."""
    from data.fred_loader import fetch_yield_curve
    df = fetch_yield_curve()
    curve = pd.Series(
        df["rate_pct"].values / 100.0,
        index=pd.Index(df["tenor_yr"].values, dtype=float),
        name="rate_decimal",
    ).sort_index()
    obs_dates = df["obs_date"].dropna()
    obs_date_str = (
        obs_dates.max().strftime("%Y-%m-%d") if not obs_dates.empty else "Fallback"
    )
    is_fallback = obs_dates.empty
    return {"curve": curve, "obs_date": obs_date_str, "is_fallback": is_fallback}


@st.cache_resource
def _load_balance_sheet():
    from models.balance_sheet import BalanceSheet
    return BalanceSheet.from_config()


@st.cache_data(ttl=86_400, show_spinner=False)
def _build_all_scenarios(cache_bust: int = 0) -> dict:
    from scenarios.rate_scenarios import build_scenarios
    meta = _load_curve_data()
    return build_scenarios(meta["curve"])


@st.cache_data(ttl=86_400, show_spinner=False)
def _compute_eve_table(cache_bust: int = 0) -> pd.DataFrame:
    from models.risk_metrics import eve_sensitivity_table
    bs = _load_balance_sheet()
    scenarios = _build_all_scenarios(cache_bust)
    return eve_sensitivity_table(bs, scenarios)


@st.cache_data(ttl=86_400, show_spinner=False)
def _compute_nii_table(cache_bust: int = 0) -> pd.DataFrame:
    from models.risk_metrics import nii_sensitivity_table
    bs = _load_balance_sheet()
    scenarios = _build_all_scenarios(cache_bust)
    return nii_sensitivity_table(bs, scenarios)


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def _get_active_scenario(all_scenarios: dict, key: str, custom_bp: dict | None):
    """Return the active RateScenario; builds custom scenario on the fly."""
    if key == "custom" and custom_bp:
        from scenarios.rate_scenarios import apply_custom_scenario, RateScenario
        meta = _load_curve_data()
        curve = meta["curve"]
        shocked = apply_custom_scenario(curve, custom_bp)
        return RateScenario(
            name="custom", label="Custom",
            base_curve=curve, shocked_curve=shocked,
        )
    return all_scenarios.get(key, all_scenarios["base"])


def _nii_breakdown_for_scenario(bs, scenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-instrument and per-bucket instantaneous NII delta for any scenario type.
    Uses the shocked curve to derive the rate change at each repricing tenor.
    """
    from models.repricing import _repricing_events, _assign_bucket, BUCKET_LABELS
    from models.risk_metrics import _interpolate_rate

    H = 12.0
    rows: list[dict] = []
    for instr in bs.assets + bs.liabilities:
        events = _repricing_events(instr, int(H))
        if not events:
            delta = 0.0
        else:
            t_yr = events[0] / 12.0
            shock = (
                _interpolate_rate(scenario.shocked_curve, t_yr)
                - _interpolate_rate(scenario.base_curve,    t_yr)
            )
            sign = 1.0 if instr.side == "asset" else -1.0
            delta = sign * instr.notional * shock * (H - events[0]) / H

        rows.append({
            "side":      instr.side,
            "name":      instr.name,
            "bucket":    _assign_bucket(instr.repricing_tenor_months),
            "reprices":  bool(events),
            "delta_nii": delta,
        })

    by_instr = pd.DataFrame(rows)
    asset_d = (
        by_instr[by_instr["side"] == "asset"]
        .groupby("bucket", sort=False)["delta_nii"].sum()
        .reindex(BUCKET_LABELS, fill_value=0.0)
    )
    liab_d = (
        by_instr[by_instr["side"] == "liability"]
        .groupby("bucket", sort=False)["delta_nii"].sum()
        .reindex(BUCKET_LABELS, fill_value=0.0)
    )
    by_bucket = pd.DataFrame({
        "bucket":          BUCKET_LABELS,
        "asset_delta":     asset_d.values,
        "liability_delta": liab_d.values,
        "net_delta":       (asset_d + liab_d).values,
    })
    return by_instr, by_bucket


def _ramp_nii_for_scenario(bs, scenario) -> float:
    """Ramp (linear) NII delta for any scenario type using shocked yield curve."""
    from models.repricing import _repricing_events
    from models.risk_metrics import _interpolate_rate

    H = 12.0
    total = 0.0
    for instr in bs.assets + bs.liabilities:
        events = _repricing_events(instr, int(H))
        if not events:
            continue
        sign = 1.0 if instr.side == "asset" else -1.0
        delta = 0.0
        for k, t_k in enumerate(events):
            t_next = events[k + 1] if k + 1 < len(events) else H
            t_yr = t_k / 12.0
            shock_full = (
                _interpolate_rate(scenario.shocked_curve, t_yr)
                - _interpolate_rate(scenario.base_curve,    t_yr)
            )
            shock_ramp = shock_full * (t_k / H)
            hold_frac  = (min(t_next, H) - t_k) / H
            delta += instr.notional * shock_ramp * hold_frac
        total += sign * delta
    return total


# ---------------------------------------------------------------------------
# Chart helpers — shared layout defaults
# ---------------------------------------------------------------------------

def _layout(**overrides) -> dict:
    base = dict(
        plot_bgcolor  = _C["bg"],
        paper_bgcolor = _C["bg"],
        font          = dict(family="Inter, Arial, sans-serif", size=12, color="#333"),
        margin        = dict(t=55, b=50, l=60, r=40),
        hovermode     = "x unified",
    )
    base.update(overrides)
    return base


def _axis(title: str = "", **kw) -> dict:
    return dict(title=title, showgrid=True, gridcolor=_C["grid"], **kw)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def fig_yield_curves(base: pd.Series, shocked: pd.Series, label: str) -> go.Figure:
    """Overlay base and shocked yield curves with a shaded spread band."""
    tenors = base.index.tolist()
    tlabels = [_TENOR_LABELS.get(t, f"{t:.1f}Y") for t in tenors]
    bp  = (base.values    * 100).tolist()
    sp  = (shocked.values * 100).tolist()

    up = any(s > b + 1e-6 for s, b in zip(sp, bp))
    fill_color = "rgba(198,40,40,0.08)" if up else "rgba(46,125,50,0.08)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tenors + tenors[::-1], y=sp + bp[::-1],
        fill="toself", fillcolor=fill_color,
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=tenors, y=bp, mode="lines+markers", name="Base",
        line=dict(color=_C["navy"], width=2.5),
        marker=dict(size=7),
        hovertemplate="Base: %{y:.3f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=tenors, y=sp, mode="lines+markers", name=f"Shocked ({label})",
        line=dict(color=_C["red"], width=2.5, dash="dot"),
        marker=dict(size=7),
        hovertemplate="Shocked: %{y:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        **_layout(hovermode="x unified"),
        title=dict(text="Yield Curve — Base vs Shocked", x=0.5),
        xaxis=dict(**_axis("Tenor"), tickvals=tenors, ticktext=tlabels),
        yaxis=_axis("Yield (%)"),
        legend=dict(x=0.65, y=0.05, bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


def fig_shock_profile(base: pd.Series, shocked: pd.Series) -> go.Figure:
    """Bar chart of shock magnitude (bp) per tenor."""
    tenors  = base.index.tolist()
    tlabels = [_TENOR_LABELS.get(t, f"{t:.1f}Y") for t in tenors]
    shocks  = ((shocked.values - base.values) * 10_000).tolist()
    colors  = [_C["red"] if s > 0 else (_C["green"] if s < 0 else _C["gray"]) for s in shocks]

    fig = go.Figure(go.Bar(
        x=tenors, y=shocks,
        marker_color=colors,
        text=[f"{s:+.0f}" for s in shocks],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Shock: %{y:+.1f} bp<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=_C["gray"])
    fig.update_layout(
        **_layout(hovermode="x"),
        title=dict(text="Shock Profile (bp per Tenor)", x=0.5),
        xaxis=dict(**_axis("Tenor"), tickvals=tenors, ticktext=tlabels),
        yaxis=_axis("Shock (bp)"),
    )
    return fig


def fig_repricing_gap(gap_df: pd.DataFrame) -> go.Figure:
    """Grouped bar: asset vs liability notional + gap per bucket."""
    buckets   = gap_df["bucket"].tolist()
    asset_n   = gap_df["asset_notional"].tolist()
    liab_n    = gap_df["liability_notional"].tolist()
    gap_vals  = gap_df["gap"].tolist()
    gap_clr   = [_C["green"] if g >= 0 else _C["red"] for g in gap_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Assets", x=buckets, y=asset_n,
        marker_color=_C["blue"], opacity=0.85,
        hovertemplate="Assets: $%{y:.1f}M<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Liabilities", x=buckets, y=liab_n,
        marker_color=_C["red"], opacity=0.85,
        hovertemplate="Liabilities: $%{y:.1f}M<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Gap (A−L)", x=buckets, y=gap_vals,
        marker_color=gap_clr, opacity=0.75,
        hovertemplate="Gap: $%{y:+.1f}M<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=_C["gray"])
    fig.update_layout(
        **_layout(),
        title=dict(text="Repricing Gap by Time Bucket ($M)", x=0.5),
        barmode="group",
        xaxis=_axis("Repricing Bucket"),
        yaxis=_axis("Notional ($M)"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


def fig_nii_by_bucket(by_bucket: pd.DataFrame, label: str) -> go.Figure:
    """Grouped bar: asset ΔNII, liability ΔNII, net ΔNII per bucket."""
    buckets = by_bucket["bucket"].tolist()
    net     = by_bucket["net_delta"].tolist()
    net_clr = [_C["green"] if v >= 0 else _C["red"] for v in net]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Asset ΔNII", x=buckets, y=by_bucket["asset_delta"].tolist(),
        marker_color=_C["blue"], opacity=0.85,
        hovertemplate="Asset ΔNII: $%{y:+.4f}M<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Liability ΔNII", x=buckets, y=by_bucket["liability_delta"].tolist(),
        marker_color=_C["red"], opacity=0.85,
        hovertemplate="Liability ΔNII: $%{y:+.4f}M<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Net ΔNII", x=buckets, y=net,
        marker_color=net_clr, opacity=0.75,
        hovertemplate="Net ΔNII: $%{y:+.4f}M<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=_C["gray"])
    fig.update_layout(
        **_layout(),
        title=dict(text=f"NII Impact by Repricing Bucket — {label}", x=0.5),
        barmode="group",
        xaxis=_axis("Repricing Bucket"),
        yaxis=_axis("ΔNII ($M)"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
    )
    return fig


def fig_nii_sensitivity(nii_table: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of ΔNII across all non-base scenarios."""
    df = nii_table[nii_table["scenario"] != "base"].copy()
    df = df.sort_values("delta_nii", ascending=True)
    clrs = [_C["green"] if v >= 0 else _C["red"] for v in df["delta_nii"]]

    fig = go.Figure(go.Bar(
        x=df["delta_nii"].tolist(), y=df["label"].tolist(),
        orientation="h", marker_color=clrs,
        text=[f"${v:+.3f}M" for v in df["delta_nii"]],
        textposition="outside",
        hovertemplate="%{y}<br>ΔNII: $%{x:+.3f}M<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color=_C["gray"])
    fig.update_layout(
        **_layout(hovermode="y", margin=dict(t=55, b=50, l=260, r=90)),
        title=dict(text="NII Sensitivity — All Scenarios", x=0.5),
        xaxis=_axis("ΔNII ($M)"),
        yaxis=dict(title=""),
    )
    return fig


def fig_eve_scenarios(eve_table: pd.DataFrame) -> go.Figure:
    """Horizontal bar of ΔEVE for all non-base scenarios."""
    df = eve_table[eve_table["scenario"] != "base"].copy()
    df = df.sort_values("delta_eve", ascending=True)
    clrs = [_C["green"] if v >= 0 else _C["red"] for v in df["delta_eve"]]

    fig = go.Figure(go.Bar(
        x=df["delta_eve"].tolist(), y=df["label"].tolist(),
        orientation="h", marker_color=clrs,
        text=[f"${v:+.2f}M" for v in df["delta_eve"]],
        textposition="outside",
        hovertemplate="%{y}<br>ΔEVE: $%{x:+.2f}M<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color=_C["gray"])
    fig.update_layout(
        **_layout(hovermode="y", margin=dict(t=55, b=50, l=260, r=90)),
        title=dict(text="ΔEVE by Scenario ($M)", x=0.5),
        xaxis=_axis("ΔEVE ($M)"),
        yaxis=dict(title=""),
    )
    return fig


def fig_duration_bars(da: float, dl: float, dg: float) -> go.Figure:
    """Bar chart of asset duration, liability duration, and duration gap."""
    labels = ["Asset Duration", "Liability Duration", "Duration Gap"]
    values = [da, dl, dg]
    colors = [_C["blue"], _C["red"], _C["green"] if dg >= 0 else _C["red"]]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.3f} yr" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.3f} years<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=_C["gray"])
    fig.update_layout(
        **_layout(hovermode="x"),
        title=dict(text="Duration Analysis (Years)", x=0.5),
        showlegend=False,
        xaxis=dict(title=""),
        yaxis=_axis("Duration (years)"),
    )
    return fig


def fig_risk_heatmap(eve_table: pd.DataFrame, nii_table: pd.DataFrame) -> go.Figure:
    """Diverging heatmap: scenarios × {ΔEVE, ΔEVE%, ΔNII, ΔNII%}."""
    eve = eve_table[eve_table["scenario"] != "base"].set_index("label")
    nii = nii_table[nii_table["scenario"] != "base"].set_index("label")

    # Sort: parallels first, non-parallel second
    all_labels = list(eve.index)
    parallel   = [l for l in all_labels if "Parallel" in l]
    non_par    = [l for l in all_labels if "Parallel" not in l]
    labels     = parallel + non_par

    cols = {
        "ΔEVE ($M)":  eve.reindex(labels)["delta_eve"].values,
        "ΔEVE (%)":   eve.reindex(labels)["pct_change_eve"].values,
        "ΔNII ($M)":  nii.reindex(labels)["delta_nii"].values,
        "ΔNII (%)":   nii.reindex(labels)["pct_change"].values,
    }
    z = np.column_stack(list(cols.values()))
    text = [[f"{v:+.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=list(cols.keys()), y=labels,
        colorscale="RdYlGn", zmid=0,
        text=text, texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.2f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="", thickness=12, len=0.75),
    ))
    fig.update_layout(
        **_layout(margin=dict(t=60, b=50, l=280, r=80), hovermode=False),
        title=dict(text="Risk Sensitivity Heatmap", x=0.5),
        xaxis=dict(title="", side="top"),
        yaxis=dict(title="", autorange="reversed"),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Page 1 — Yield Curve
# ---------------------------------------------------------------------------

def page_yield_curve(scenario, meta: dict) -> None:
    st.header("Yield Curve")
    st.caption(
        "US Treasury yield curve (latest FRED observation). "
        "The shocked curve shows how the selected scenario shifts each tenor."
    )

    if meta["is_fallback"]:
        st.warning("FRED API unavailable — showing hardcoded fallback curve.", icon="⚠️")
    else:
        st.caption(f"Data as of **{meta['obs_date']}**")

    base    = scenario.base_curve
    shocked = scenario.shocked_curve

    # KPI row
    interp2y_base = float(np.interp(2.0,  base.index, base.values))
    interp10_base = float(np.interp(10.0, base.index, base.values))
    shock_2y_bp   = (float(np.interp(2.0,  shocked.index, shocked.values)) - interp2y_base)  * 10_000
    shock_10y_bp  = (float(np.interp(10.0, shocked.index, shocked.values)) - interp10_base) * 10_000

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("2Y Treasury",  f"{interp2y_base*100:.3f}%",
              help="Current 2-year CMT yield (base curve).")
    c2.metric("10Y Treasury", f"{interp10_base*100:.3f}%",
              help="Current 10-year CMT yield (base curve).")
    c3.metric("Shock @ 2Y",  f"{shock_2y_bp:+.0f} bp",
              help="Net rate change at the 2-year tenor in the selected scenario.")
    c4.metric("Shock @ 10Y", f"{shock_10y_bp:+.0f} bp",
              help="Net rate change at the 10-year tenor in the selected scenario.")

    st.divider()

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.plotly_chart(fig_yield_curves(base, shocked, scenario.label), use_container_width=True)
        st.plotly_chart(fig_shock_profile(base, shocked), use_container_width=True)

    with col_table:
        st.subheader("Rate Table")
        tenors    = base.index.tolist()
        shock_bps = ((shocked.values - base.values) * 10_000).tolist()
        rate_df   = pd.DataFrame({
            "Tenor":         [_TENOR_LABELS.get(t, f"{t:.2f}Y") for t in tenors],
            "Base (%)":      [f"{v*100:.3f}" for v in base.values],
            "Shocked (%)":   [f"{v*100:.3f}" for v in shocked.values],
            "Shock (bp)":    [f"{v:+.1f}"    for v in shock_bps],
        })
        st.dataframe(rate_df, use_container_width=True, hide_index=True)

        st.info(
            "**BCBS/EBA IRRBB calibration:** Non-parallel scenarios use a short-end "
            "plateau (≤ 2Y), linear transition (2Y–10Y), and long-end plateau (≥ 10Y). "
            "Parallel scenarios apply a uniform shift across all tenors.",
            icon="ℹ️",
        )


# ---------------------------------------------------------------------------
# Page 2 — Balance Sheet
# ---------------------------------------------------------------------------

def page_balance_sheet(bs) -> None:
    st.header("Balance Sheet")
    st.caption(
        "Synthetic bank balance sheet with assets, liabilities, and repricing profile. "
        "All values in $M. Repricing buckets follow IRRBB standard boundaries (<3M, 3M–1Y, 1Y–5Y, >5Y)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Assets",      f"${bs.total_assets:,.0f}M",
              help="Sum of all asset notionals.")
    c2.metric("Total Liabilities", f"${bs.total_liabilities:,.0f}M",
              help="Sum of all liability notionals.")
    c3.metric("Equity",            f"${bs.equity:,.0f}M",
              help="Regulatory capital (not interest-bearing).")
    c4.metric("Base NII (ann.)",   f"${bs.net_interest_income:,.2f}M",
              help="Annualised NII = Σ(asset income) − Σ(liability cost) at current coupons.")
    st.caption(f"Net Interest Margin: **{bs.net_interest_margin:.3f}%**")

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Instruments")
        df = bs.to_dataframe()
        display_df = df[[
            "side", "name", "notional", "rate_type",
            "coupon_pct", "repricing_tenor_months",
            "maturity_years", "repricing_bucket", "annual_income_$M",
        ]].copy()
        display_df.columns = [
            "Side", "Instrument", "Notional ($M)", "Rate Type",
            "Coupon (%)", "Repricing (M)", "Maturity (Y)", "Bucket", "Ann. Inc. ($M)",
        ]
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Notional ($M)":  st.column_config.NumberColumn(format="$%.1f"),
                "Coupon (%)":     st.column_config.NumberColumn(format="%.3f%%"),
                "Ann. Inc. ($M)": st.column_config.NumberColumn(format="$%.3f"),
                "Repricing (M)":  st.column_config.NumberColumn(format="%.0f M"),
            },
        )

        st.subheader("Repricing Gap Summary (4-Bucket IRRBB)")
        gap_sum = bs.repricing_summary()
        gap_sum.columns = [
            "Bucket", "Assets ($M)", "Assets (%)",
            "Liabilities ($M)", "Liabilities (%)", "Gap ($M)", "Cum. Gap ($M)",
        ]
        st.dataframe(
            gap_sum,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Assets ($M)":       st.column_config.NumberColumn(format="$%.1f"),
                "Liabilities ($M)":  st.column_config.NumberColumn(format="$%.1f"),
                "Gap ($M)":          st.column_config.NumberColumn(format="$%+.1f"),
                "Cum. Gap ($M)":     st.column_config.NumberColumn(format="$%+.1f"),
                "Assets (%)":        st.column_config.NumberColumn(format="%.1f%%"),
                "Liabilities (%)":   st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    with col_right:
        from models.repricing import build_repricing_gap
        gap7 = build_repricing_gap(bs)
        st.plotly_chart(fig_repricing_gap(gap7), use_container_width=True)
        st.info(
            "**Positive gap** (asset-heavy): rising rates increase NII in that bucket. "
            "**Negative gap** (liability-heavy): rising rates compress NII. "
            "This bank has a large negative gap in the <3M bucket (floating liabilities dominate).",
            icon="ℹ️",
        )


# ---------------------------------------------------------------------------
# Page 3 — NII Impact
# ---------------------------------------------------------------------------

def page_nii(bs, scenario, nii_table: pd.DataFrame) -> None:
    st.header("NII Impact")
    st.caption(
        "Net Interest Income (NII) sensitivity to the selected rate shock over a **12-month horizon**. "
        "Only instruments that reprice within the horizon contribute to ΔNII."
    )

    _, by_bucket     = _nii_breakdown_for_scenario(bs, scenario)
    instant_delta    = float(by_bucket["net_delta"].sum())
    ramp_delta       = _ramp_nii_for_scenario(bs, scenario)
    base_nii         = bs.net_interest_income
    pct_inst         = instant_delta / abs(base_nii) * 100 if base_nii else 0.0
    pct_ramp         = ramp_delta    / abs(base_nii) * 100 if base_nii else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base NII",             f"${base_nii:,.3f}M",
              help="Annualised NII at current coupon rates.")
    c2.metric("ΔNII — Instantaneous", f"${instant_delta:+,.3f}M",
              delta=f"{pct_inst:+.2f}%", delta_color="normal",
              help="Full shock applied at t=0; instruments earn the shocked rate after repricing.")
    c3.metric("ΔNII — Ramp (12M)",    f"${ramp_delta:+,.3f}M",
              delta=f"{pct_ramp:+.2f}%", delta_color="normal",
              help="Shock builds linearly from 0 to peak over 12 months. "
                   "Typically ~50% of the instantaneous impact.")
    c4.metric("Shocked NII",          f"${base_nii + instant_delta:,.3f}M",
              help="Projected NII under instantaneous shock.")

    st.divider()

    col_chart, col_side = st.columns([3, 2])

    with col_chart:
        st.plotly_chart(
            fig_nii_by_bucket(by_bucket, scenario.label),
            use_container_width=True,
        )
        st.caption(
            "**Blue**: asset repricing contribution (income gain = positive). "
            "**Red**: liability repricing contribution (cost increase = negative). "
            "**Net bar**: combined impact per time bucket."
        )

    with col_side:
        st.subheader("Bucket Detail")
        bkt_disp = by_bucket.copy()
        bkt_disp.columns = ["Bucket", "Asset ΔNII", "Liab. ΔNII", "Net ΔNII"]
        st.dataframe(
            bkt_disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Asset ΔNII": st.column_config.NumberColumn(format="$%+.4f"),
                "Liab. ΔNII": st.column_config.NumberColumn(format="$%+.4f"),
                "Net ΔNII":   st.column_config.NumberColumn(format="$%+.4f"),
            },
        )

        st.subheader("Instantaneous vs Ramp")
        cmp = pd.DataFrame([
            {
                "Method":       "Instantaneous",
                "ΔNII ($M)":    round(instant_delta, 4),
                "ΔNII (%)":     round(pct_inst, 3),
                "Shocked NII":  round(base_nii + instant_delta, 4),
            },
            {
                "Method":       "Ramp (12M)",
                "ΔNII ($M)":    round(ramp_delta, 4),
                "ΔNII (%)":     round(pct_ramp, 3),
                "Shocked NII":  round(base_nii + ramp_delta, 4),
            },
        ])
        st.dataframe(
            cmp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ΔNII ($M)":   st.column_config.NumberColumn(format="$%+.4f"),
                "ΔNII (%)":    st.column_config.NumberColumn(format="%+.3f%%"),
                "Shocked NII": st.column_config.NumberColumn(format="$%.4f"),
            },
        )
        st.caption(
            "The ramp scenario is typically ~50% of the instantaneous impact because "
            "the shock builds gradually, so earlier repricing events lock in only a "
            "partial rate change."
        )

    st.divider()
    st.subheader("NII Sensitivity — All Scenarios")
    col_a, col_b = st.columns([2, 3])
    with col_a:
        nii_tbl_disp = nii_table[["label", "nii", "delta_nii", "pct_change"]].copy()
        nii_tbl_disp.columns = ["Scenario", "NII ($M)", "ΔNII ($M)", "ΔNII (%)"]
        st.dataframe(
            nii_tbl_disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "NII ($M)":  st.column_config.NumberColumn(format="$%.3f"),
                "ΔNII ($M)": st.column_config.NumberColumn(format="$%+.3f"),
                "ΔNII (%)":  st.column_config.NumberColumn(format="%+.2f%%"),
            },
        )
    with col_b:
        st.plotly_chart(fig_nii_sensitivity(nii_table), use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4 — EVE Impact
# ---------------------------------------------------------------------------

def page_eve(bs, scenario, eve_table: pd.DataFrame) -> None:
    st.header("EVE Impact")
    st.caption(
        "**Economic Value of Equity (EVE)** = PV(Assets) − PV(Liabilities), discounted using the shocked yield curve. "
        "Fixed-rate instruments lose market value when rates rise; floating instruments "
        "reprice at par at their next reset date (repricing-at-par assumption)."
    )

    from models.risk_metrics import compute_eve
    with st.spinner("Computing EVE for selected scenario…"):
        result = compute_eve(bs, scenario)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base EVE",        f"${result.base_eve:,.2f}M",
              help="EVE under the base (unshocked) yield curve.")
    c2.metric("ΔEVE (Shocked)",  f"${result.delta_eve:+,.2f}M",
              delta=f"{result.pct_change_eve:+.2f}% of |base EVE|",
              delta_color="normal",
              help="Change in EVE under the selected rate shock.")
    c3.metric("Duration Gap",    f"{result.duration_gap:+.3f} yr",
              help="Duration Gap = DA − (PV_L / PV_A) × DL.\n"
                   "Positive = asset-sensitive: EVE falls when rates rise.")
    c4.metric("BP01",            f"${result.bp01:+.4f}M/bp",
              help="EVE change per +1 bp parallel shift on the base curve (central difference).")

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("ΔEVE Across All Scenarios")
        st.plotly_chart(fig_eve_scenarios(eve_table), use_container_width=True)
        st.caption(
            "**Green** (positive ΔEVE): rate decreases cause long fixed-rate assets to appreciate. "
            "**Red** (negative ΔEVE): rate increases erode the value of long-duration assets. "
            "The Bull Steepener and Bull Flattener produce the largest EVE gains because long rates fall most."
        )

        st.subheader("EVE Sensitivity Table — All Scenarios")
        eve_disp = eve_table[[
            "label", "pv_assets", "pv_liabilities", "eve",
            "delta_eve", "pct_change_eve", "duration_gap", "bp01",
        ]].copy()
        eve_disp.columns = [
            "Scenario", "PV Assets ($M)", "PV Liabs ($M)", "EVE ($M)",
            "ΔEVE ($M)", "ΔEVE (%)", "Dur. Gap (Y)", "BP01 ($M/bp)",
        ]
        st.dataframe(
            eve_disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "PV Assets ($M)":  st.column_config.NumberColumn(format="$%.2f"),
                "PV Liabs ($M)":   st.column_config.NumberColumn(format="$%.2f"),
                "EVE ($M)":        st.column_config.NumberColumn(format="$%.2f"),
                "ΔEVE ($M)":       st.column_config.NumberColumn(format="$%+.2f"),
                "ΔEVE (%)":        st.column_config.NumberColumn(format="%+.2f%%"),
                "Dur. Gap (Y)":    st.column_config.NumberColumn(format="%+.3f"),
                "BP01 ($M/bp)":    st.column_config.NumberColumn(format="$%+.4f"),
            },
        )

    with col_right:
        st.subheader("Duration Analysis")
        st.plotly_chart(
            fig_duration_bars(result.duration_assets, result.duration_liabilities, result.duration_gap),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        c1.metric("Asset Duration",     f"{result.duration_assets:.3f} yr",
                  help="PV-weighted average modified duration of all assets.")
        c2.metric("Liability Duration", f"{result.duration_liabilities:.3f} yr",
                  help="PV-weighted average modified duration of all liabilities.")

        st.info(
            "A positive **Duration Gap** means the bank's assets have longer "
            "effective duration than its liabilities. Rising rates therefore reduce EVE "
            "(asset losses > liability gains). This is the typical position for retail banks "
            "with long-dated mortgages funded by short-term deposits.",
            icon="ℹ️",
        )

    st.divider()
    st.subheader("Per-Instrument EVE Breakdown")
    st.caption(
        f"Present values and durations under the **base** and **shocked** ({scenario.label}) curves."
    )
    instr_df = result.by_instrument[[
        "side", "name", "notional", "rate_type", "coupon_pct",
        "effective_maturity_yr", "pv_base", "pv_shocked",
        "delta_pv", "delta_pv_pct", "modified_duration", "dv01",
    ]].copy()
    instr_df.columns = [
        "Side", "Instrument", "Notional ($M)", "Rate Type", "Coupon (%)",
        "Mat. (Y)", "PV Base ($M)", "PV Shocked ($M)",
        "ΔPV ($M)", "ΔPV (%)", "Mod. Dur.", "DV01 ($M)",
    ]
    st.dataframe(
        instr_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Notional ($M)":    st.column_config.NumberColumn(format="$%.1f"),
            "Coupon (%)":       st.column_config.NumberColumn(format="%.3f%%"),
            "PV Base ($M)":     st.column_config.NumberColumn(format="$%.3f"),
            "PV Shocked ($M)":  st.column_config.NumberColumn(format="$%.3f"),
            "ΔPV ($M)":         st.column_config.NumberColumn(format="$%+.3f"),
            "ΔPV (%)":          st.column_config.NumberColumn(format="%+.2f%%"),
            "Mod. Dur.":        st.column_config.NumberColumn(format="%.3f"),
            "DV01 ($M)":        st.column_config.NumberColumn(format="$%+.5f"),
        },
    )


# ---------------------------------------------------------------------------
# Page 5 — Risk Summary
# ---------------------------------------------------------------------------

def page_risk_summary(bs, eve_table: pd.DataFrame, nii_table: pd.DataFrame) -> None:
    st.header("Risk Summary")
    st.caption(
        "Combined IRRBB risk snapshot across all 8 regulatory scenarios. "
        "Designed for senior management and regulatory reporting."
    )

    # Extract base-case duration/BP01 from eve_table
    base_row = eve_table[eve_table["scenario"] == "base"]
    if base_row.empty:
        st.error("Base scenario not found in EVE table.")
        return
    base_row = base_row.iloc[0]
    dg       = float(base_row["duration_gap"])
    bp01     = float(base_row["bp01"])
    base_nii = bs.net_interest_income

    non_base_eve = eve_table[eve_table["scenario"] != "base"]
    non_base_nii = nii_table[nii_table["scenario"] != "base"]
    nii_at_risk  = float(non_base_nii["delta_nii"].min())
    eve_at_risk  = float(non_base_eve["delta_eve"].min())
    nii_upside   = float(non_base_nii["delta_nii"].max())
    eve_upside   = float(non_base_eve["delta_eve"].max())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Duration Gap",   f"{dg:+.3f} yr",
              help="Positive = asset-sensitive. Rising rates decrease EVE.")
    c2.metric("BP01",           f"${bp01:+.4f}M/bp",
              help="EVE change per 1 bp parallel shift (central difference on base curve).")
    c3.metric("NII-at-Risk",    f"${nii_at_risk:+.3f}M",
              delta=f"{nii_at_risk/abs(base_nii)*100:+.2f}%",
              delta_color="inverse",
              help="Worst-case ΔNII across the 8 standard IRRBB scenarios (instantaneous).")
    c4.metric("EVE-at-Risk",    f"${eve_at_risk:+.2f}M",
              delta=f"Upside: ${eve_upside:+.2f}M",
              delta_color="off",
              help="Worst-case ΔEVE across the 8 standard IRRBB scenarios.")
    c5.metric("Base NIM",       f"{bs.net_interest_margin:.3f}%",
              help="Net Interest Margin = NII / Total Assets.")

    st.divider()

    st.subheader("Sensitivity Heatmap — All Scenarios × Key Risk Metrics")
    st.plotly_chart(fig_risk_heatmap(eve_table, nii_table), use_container_width=True)
    st.caption(
        "**Green** = beneficial impact (EVE/NII improve). **Red** = adverse impact. "
        "Values shown: ΔEVE in $M and %, ΔNII in $M and %. "
        "The bank is asset-sensitive: parallel rate rises hurt EVE but mildly lift NII."
    )

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("ΔNII by Scenario")
        df_n = non_base_nii.copy()
        df_n["color"] = df_n["delta_nii"].apply(lambda v: "Positive" if v >= 0 else "Negative")
        fig_n = px.bar(
            df_n.sort_values("delta_nii"),
            x="delta_nii", y="label", orientation="h",
            color="color",
            color_discrete_map={"Positive": _C["green"], "Negative": _C["red"]},
            labels={"delta_nii": "ΔNII ($M)", "label": ""},
            text=[f"${v:+.3f}M" for v in df_n.sort_values("delta_nii")["delta_nii"]],
        )
        fig_n.update_traces(textposition="outside")
        fig_n.update_layout(
            **_layout(hovermode="y", margin=dict(t=40, b=40, l=260, r=90)),
            showlegend=False,
        )
        fig_n.add_vline(x=0, line_color=_C["gray"], line_width=1)
        st.plotly_chart(fig_n, use_container_width=True)

    with col_r:
        st.subheader("ΔEVE by Scenario")
        df_e = non_base_eve.copy()
        df_e["color"] = df_e["delta_eve"].apply(lambda v: "Positive" if v >= 0 else "Negative")
        fig_e = px.bar(
            df_e.sort_values("delta_eve"),
            x="delta_eve", y="label", orientation="h",
            color="color",
            color_discrete_map={"Positive": _C["green"], "Negative": _C["red"]},
            labels={"delta_eve": "ΔEVE ($M)", "label": ""},
            text=[f"${v:+.2f}M" for v in df_e.sort_values("delta_eve")["delta_eve"]],
        )
        fig_e.update_traces(textposition="outside")
        fig_e.update_layout(
            **_layout(hovermode="y", margin=dict(t=40, b=40, l=260, r=90)),
            showlegend=False,
        )
        fig_e.add_vline(x=0, line_color=_C["gray"], line_width=1)
        st.plotly_chart(fig_e, use_container_width=True)

    st.divider()
    st.subheader("IRRBB Regulatory Commentary")
    direction = "asset-sensitive" if dg > 0 else "liability-sensitive"
    eve_pct_worst = float(non_base_eve["pct_change_eve"].min())

    with st.expander("Risk Assessment & Regulatory Context", expanded=True):
        st.markdown(f"""
**Interest Rate Risk Position: {direction.upper()}**

This bank has a Duration Gap of **{dg:+.3f} years**, meaning its assets have longer effective
duration than its liabilities. As a result:

- **EVE** is adversely affected by **rising rates** (long fixed-rate assets lose market value
  faster than short-maturity liabilities)
- **NII** is mildly supported by rising rates in the short term (floating assets and liabilities
  reprice within the 12-month horizon, with a modest asset-side advantage)

---

| Metric | Value | Signal |
|--------|-------|--------|
| Duration Gap | {dg:+.3f} yr | {"Asset-sensitive ↑" if dg > 0 else "Liability-sensitive ↓"} |
| BP01 | ${bp01:+.4f}M per bp | EVE sensitivity to parallel shift |
| NII-at-Risk | ${nii_at_risk:+.3f}M ({nii_at_risk/abs(base_nii)*100:+.2f}%) | Worst-case 12M NII change |
| EVE-at-Risk | ${eve_at_risk:+.2f}M ({eve_pct_worst:+.2f}%) | Worst-case EVE change |
| NII Upside | ${nii_upside:+.3f}M | Best-case 12M NII change |
| EVE Upside | ${eve_upside:+.2f}M | Best-case EVE change |

---

**Regulatory Reference (EBA IRRBB Guidelines)**

Under EBA/GL/2018/02 and Article 98(5) CRD V, supervisors apply an **EVE outlier test**:
banks where ΔEVE under any standard scenario exceeds **15% of Tier 1 capital** are
flagged for supervisory review. The NII outlier threshold is **≥ 5% of projected NII**
over a 12-month horizon. Banks should also maintain adequate capital buffers against
the identified worst-case scenario.
        """)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(all_scenarios: dict, meta: dict) -> tuple[str, str, dict | None, bool]:
    """Render sidebar; return (page, scenario_key, custom_shocks_bp, refresh_clicked)."""
    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center;padding:10px 0 4px;"
            f"color:{_C['navy']};font-size:1.2rem;font-weight:700'>"
            f"NII / EVE Simulator</div>",
            unsafe_allow_html=True,
        )
        st.caption("Interest Rate Risk in the Banking Book")
        st.divider()

        # Page navigation
        page = st.radio(
            "Page",
            options=["Yield Curve", "Balance Sheet", "NII Impact", "EVE Impact", "Risk Summary"],
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("Scenario")

        # Build label → key mapping
        scenario_labels = [sc.label for sc in all_scenarios.values()] + ["Custom"]
        scenario_keys   = list(all_scenarios.keys()) + ["custom"]
        label_to_key    = dict(zip(scenario_labels, scenario_keys))

        selected_label = st.selectbox(
            "Rate Scenario",
            options=scenario_labels,
            index=0,
            help=(
                "Select the IRRBB rate shock scenario. "
                "Affects the Yield Curve, NII Impact, and EVE Impact pages."
            ),
        )
        selected_key = label_to_key[selected_label]

        # Custom tenor sliders
        custom_shocks_bp: dict | None = None
        if selected_key == "custom":
            st.caption("Shock (bp) per tenor — interpolated between points:")
            custom_shocks_bp = {}
            for tenor, tlabel in _SLIDER_TENORS:
                custom_shocks_bp[tenor] = float(st.slider(
                    tlabel, min_value=-400, max_value=400, value=0, step=25,
                    help=f"Rate shock at {tlabel} in basis points. "
                         "Shocks at other tenors are linearly interpolated.",
                ))

        st.divider()
        st.subheader("Data")

        is_fallback = meta.get("is_fallback", False)
        obs_label   = meta.get("obs_date", "unknown")
        if is_fallback:
            st.warning("Using fallback curve (FRED unavailable)", icon="⚠️")
        else:
            st.caption(f"Curve as of: **{obs_label}**")

        refresh_clicked = st.button(
            "Refresh FRED Data",
            help="Clear the on-disk CSV cache and re-fetch from FRED.",
        )
        st.caption("Source: FRED (Federal Reserve Bank of St. Louis)")

    return page, selected_key, custom_shocks_bp, refresh_clicked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="NII / EVE Simulator",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Minimal CSS tweaks
    st.markdown("""
    <style>
    [data-testid="stMetricValue"]  { font-size: 1.25rem; font-weight: 700; }
    [data-testid="stMetricLabel"]  { font-size: 0.78rem; color: #546E7A; }
    .block-container               { padding-top: 1.5rem; }
    div[data-testid="stSidebarNav"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    if "cache_bust" not in st.session_state:
        st.session_state["cache_bust"] = 0

    cb = st.session_state["cache_bust"]

    # Load data
    with st.spinner("Loading yield curve…"):
        meta = _load_curve_data()

    bs = _load_balance_sheet()

    with st.spinner("Building scenarios…"):
        all_scenarios = _build_all_scenarios(cb)

    # Render sidebar
    page, selected_key, custom_bp, refresh_clicked = render_sidebar(all_scenarios, meta)

    # Handle refresh
    if refresh_clicked:
        cache_dir = Path("data/.cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        st.cache_data.clear()
        st.session_state["cache_bust"] += 1
        st.rerun()

    # Build active scenario
    active_scenario = _get_active_scenario(all_scenarios, selected_key, custom_bp)

    # Pre-compute sensitivity tables (all scenarios, cached)
    with st.spinner("Computing scenario sensitivity tables…"):
        eve_table = _compute_eve_table(cb)
        nii_table = _compute_nii_table(cb)

    # Route
    if page == "Yield Curve":
        page_yield_curve(active_scenario, meta)
    elif page == "Balance Sheet":
        page_balance_sheet(bs)
    elif page == "NII Impact":
        page_nii(bs, active_scenario, nii_table)
    elif page == "EVE Impact":
        page_eve(bs, active_scenario, eve_table)
    elif page == "Risk Summary":
        page_risk_summary(bs, eve_table, nii_table)


if __name__ == "__main__":
    main()
