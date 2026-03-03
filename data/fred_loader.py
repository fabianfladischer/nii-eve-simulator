"""
data/fred_loader.py
-------------------
Utilities for fetching interest rate data from the FRED API.

Responsibilities:
- Authenticate with the FRED API using the key in config.py
- Fetch SOFR and US Treasury CMT yield curve series
- Parse and normalise responses into clean DataFrames (tenor in years, yield in %)
- Cache results to CSV; reload from cache if younger than 1 day
- Fall back to a hardcoded curve when the API is unavailable
- Provide a Plotly visualisation helper
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FRED series IDs and their corresponding tenor in years.
# SOFR has no tenor in the yield-curve sense; it is stored separately.
TREASURY_SERIES: dict[str, float] = {
    "DGS1MO":  1 / 12,
    "DGS3MO":  3 / 12,
    "DGS6MO":  6 / 12,
    "DGS1":    1.0,
    "DGS2":    2.0,
    "DGS5":    5.0,
    "DGS7":    7.0,
    "DGS10":  10.0,
    "DGS20":  20.0,
    "DGS30":  30.0,
}

SOFR_SERIES: str = "SOFR"

# Cache directory lives next to this file
_CACHE_DIR = Path(__file__).parent / ".cache"
_CACHE_MAX_AGE = timedelta(hours=24)

# Fallback curve: approximate values in % (as of late 2024)
_FALLBACK_CURVE: dict[float, float] = {
    1 / 12: 5.33,
    3 / 12: 5.30,
    6 / 12: 5.17,
    1.0:    4.97,
    2.0:    4.62,
    5.0:    4.40,
    7.0:    4.42,
    10.0:   4.46,
    20.0:   4.74,
    30.0:   4.63,
}
_FALLBACK_SOFR: float = 5.31


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cache_path(name: str) -> Path:
    """Return the CSV cache file path for a given name."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{name}.csv"


def _cache_is_fresh(path: Path) -> bool:
    """Return True if *path* exists and was modified within the cache max age."""
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < _CACHE_MAX_AGE


def _get_fred_client():
    """Instantiate and return a ``fredapi.Fred`` client."""
    try:
        from fredapi import Fred  # local import so the module loads without fredapi
    except ImportError as exc:
        raise ImportError(
            "fredapi is not installed. Run: pip install fredapi"
        ) from exc

    api_key = config.FRED_API_KEY
    if not api_key or api_key == "YOUR_FRED_API_KEY_HERE":
        raise ValueError(
            "FRED_API_KEY is not configured. Set it in config.py."
        )
    return Fred(api_key=api_key)


def _latest_observation(series: pd.Series) -> float | None:
    """
    Return the most recent non-NaN value from a date-indexed series,
    or None if the series is empty / all NaN.
    """
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


# ---------------------------------------------------------------------------
# Public API – individual series
# ---------------------------------------------------------------------------


def fetch_series(
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.Series:
    """
    Fetch an arbitrary FRED series as a date-indexed pandas Series.

    Results are cached to CSV for up to 24 hours.  When the API is
    unavailable, the function raises the underlying exception so callers
    can decide how to handle it.

    Parameters
    ----------
    series_id:
        FRED series identifier (e.g., ``"DGS10"``).
    start_date:
        ISO-8601 start date, inclusive.  Defaults to 30 days ago.
    end_date:
        ISO-8601 end date, inclusive.  Defaults to today.

    Returns
    -------
    pd.Series
        Date-indexed series of observations (values as-returned by FRED,
        typically percentages for rate series).
    """
    cache_key = f"{series_id}_{start_date or 'open'}_{end_date or 'today'}"
    cache_file = _cache_path(cache_key)

    if _cache_is_fresh(cache_file):
        logger.debug("Cache hit for %s (%s)", series_id, cache_file.name)
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze("columns")
        return cached

    logger.info("Fetching %s from FRED…", series_id)
    fred = _get_fred_client()

    end = end_date or datetime.today().strftime("%Y-%m-%d")
    start = start_date or (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")

    series = fred.get_series(series_id, observation_start=start, observation_end=end)

    if series is None or series.empty:
        logger.warning("FRED returned empty series for %s", series_id)
        return pd.Series(dtype=float, name=series_id)

    series.name = series_id
    series.to_csv(cache_file)
    logger.debug("Cached %s → %s", series_id, cache_file.name)

    return series


# ---------------------------------------------------------------------------
# Public API – yield curve
# ---------------------------------------------------------------------------


def fetch_yield_curve(as_of_date: str | None = None) -> pd.DataFrame:
    """
    Build a clean US Treasury yield curve DataFrame from FRED data.

    Each Treasury CMT series is fetched (with caching), the most recent
    non-NaN observation is selected, and the result is assembled into a
    tidy DataFrame.  If the API fails entirely, a hardcoded fallback curve
    is returned and a warning is logged.

    Parameters
    ----------
    as_of_date:
        ISO-8601 date string (``"YYYY-MM-DD"``).  When provided, the
        observation on or just before this date is used.  Defaults to the
        most recent available value.

    Returns
    -------
    pd.DataFrame
        Columns:

        - ``tenor_yr``   – Tenor in years (float)
        - ``rate_pct``   – Yield in percent (e.g. ``4.50`` for 4.50 %)
        - ``series_id``  – Source FRED series ID (str)
        - ``obs_date``   – Date of the observation (``pd.Timestamp``)
    """
    cache_file = _cache_path(f"yield_curve_{as_of_date or 'latest'}")

    if _cache_is_fresh(cache_file):
        logger.info("Yield curve loaded from cache (%s)", cache_file.name)
        return pd.read_csv(cache_file, parse_dates=["obs_date"])

    rows: list[dict] = []
    any_api_success = False

    end = as_of_date or datetime.today().strftime("%Y-%m-%d")
    # Look back 30 days to find the most recent observation
    start = (
        datetime.strptime(end, "%Y-%m-%d") - timedelta(days=30)
    ).strftime("%Y-%m-%d")

    for series_id, tenor_yr in TREASURY_SERIES.items():
        try:
            series = fetch_series(series_id, start_date=start, end_date=end)
            clean = series.dropna()
            if clean.empty:
                logger.warning("%s: no non-NaN observations in window; skipping", series_id)
                continue
            # Use the observation on or before as_of_date
            obs_value = float(clean.iloc[-1])
            obs_date = clean.index[-1]
            rows.append({
                "tenor_yr":  tenor_yr,
                "rate_pct":  obs_value,
                "series_id": series_id,
                "obs_date":  obs_date,
            })
            any_api_success = True
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", series_id, exc)

    if not rows:
        logger.warning(
            "All FRED requests failed – using hardcoded fallback yield curve."
        )
        return _fallback_yield_curve()

    df = (
        pd.DataFrame(rows)
        .sort_values("tenor_yr")
        .reset_index(drop=True)
    )

    # Fill any gaps (missing tenors) by linear interpolation
    n_fetched = len(df)
    n_expected = len(TREASURY_SERIES)
    if n_fetched < n_expected:
        logger.warning(
            "Yield curve has %d/%d tenors; missing points will be interpolated "
            "by downstream consumers.",
            n_fetched,
            n_expected,
        )

    df.to_csv(cache_file, index=False)
    logger.info("Yield curve fetched from FRED (%d tenors) and cached.", len(df))

    return df


def fetch_sofr(as_of_date: str | None = None) -> float:
    """
    Return the most recent SOFR rate in percent.

    Falls back to the hardcoded value if the API is unavailable.

    Parameters
    ----------
    as_of_date:
        ISO-8601 date string.  Defaults to today.

    Returns
    -------
    float
        SOFR in percent (e.g. ``5.31``).
    """
    end = as_of_date or datetime.today().strftime("%Y-%m-%d")
    start = (
        datetime.strptime(end, "%Y-%m-%d") - timedelta(days=10)
    ).strftime("%Y-%m-%d")

    try:
        series = fetch_series(SOFR_SERIES, start_date=start, end_date=end)
        value = _latest_observation(series)
        if value is None:
            raise ValueError("No SOFR observations found.")
        logger.info("SOFR = %.4f%%", value)
        return value
    except Exception as exc:
        logger.warning("SOFR fetch failed (%s) – using fallback %.4f%%", exc, _FALLBACK_SOFR)
        return _FALLBACK_SOFR


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def _fallback_yield_curve() -> pd.DataFrame:
    """Return the hardcoded fallback yield curve as a properly-structured DataFrame."""
    rows = [
        {
            "tenor_yr":  tenor,
            "rate_pct":  rate,
            "series_id": _tenor_to_series_id(tenor),
            "obs_date":  pd.NaT,
        }
        for tenor, rate in _FALLBACK_CURVE.items()
    ]
    df = pd.DataFrame(rows).sort_values("tenor_yr").reset_index(drop=True)
    logger.warning("Using fallback yield curve (hardcoded values).")
    return df


def _tenor_to_series_id(tenor_yr: float) -> str:
    """Reverse-lookup: tenor → FRED series ID."""
    for sid, t in TREASURY_SERIES.items():
        if abs(t - tenor_yr) < 1e-9:
            return sid
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Helpers / metadata
# ---------------------------------------------------------------------------


def list_available_series() -> dict[str, str]:
    """
    Return a mapping of friendly names to FRED series IDs used by the simulator.

    Returns
    -------
    dict[str, str]
        ``{"Friendly label": "FRED_SERIES_ID", ...}``
    """
    series_map = {
        "SOFR (Overnight)":           SOFR_SERIES,
        "Treasury 1-Month":           "DGS1MO",
        "Treasury 3-Month":           "DGS3MO",
        "Treasury 6-Month":           "DGS6MO",
        "Treasury 1-Year":            "DGS1",
        "Treasury 2-Year":            "DGS2",
        "Treasury 5-Year":            "DGS5",
        "Treasury 7-Year":            "DGS7",
        "Treasury 10-Year":           "DGS10",
        "Treasury 20-Year":           "DGS20",
        "Treasury 30-Year":           "DGS30",
    }
    return series_map


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_yield_curve(
    df: pd.DataFrame | None = None,
    title: str = "US Treasury Yield Curve",
    show: bool = True,
) -> go.Figure:
    """
    Plot the yield curve using Plotly.

    Parameters
    ----------
    df:
        Output of :func:`fetch_yield_curve`.  Fetches live data when *None*.
    title:
        Chart title.
    show:
        Call ``fig.show()`` before returning when True (default).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None:
        df = fetch_yield_curve()

    obs_date = df["obs_date"].dropna()
    subtitle = (
        f"As of {obs_date.max().strftime('%Y-%m-%d')}"
        if not obs_date.empty
        else "Fallback (hardcoded)"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["tenor_yr"],
            y=df["rate_pct"],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=7),
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "Tenor: %{x:.2f} yr<br>"
                "Yield: %{y:.3f}%<extra></extra>"
            ),
            customdata=df["series_id"],
            name="Yield",
        )
    )

    # Annotate each point with the series label
    tenor_labels = {
        1 / 12: "1M", 3 / 12: "3M", 6 / 12: "6M",
        1.0: "1Y", 2.0: "2Y", 5.0: "5Y",
        7.0: "7Y", 10.0: "10Y", 20.0: "20Y", 30.0: "30Y",
    }
    for _, row in df.iterrows():
        label = tenor_labels.get(row["tenor_yr"], f"{row['tenor_yr']:.2f}Y")
        fig.add_annotation(
            x=row["tenor_yr"],
            y=row["rate_pct"],
            text=label,
            showarrow=False,
            yshift=12,
            font=dict(size=10),
        )

    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>", x=0.5),
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
        margin=dict(t=80, b=60, l=60, r=40),
    )

    if show:
        fig.show()

    return fig
