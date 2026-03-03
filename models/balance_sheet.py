"""
models/balance_sheet.py
-----------------------
Balance sheet data model for the NII/EVE simulator.

Design notes
------------
- ``Instrument`` is a plain class (not a dataclass) so downstream code can
  subclass or duck-type it without import gymnastics.
- ``BalanceSheet`` owns the authoritative list of assets and liabilities and
  is the single source of truth for totals, NII and the repricing gap ladder.
- All monetary values are in $M (millions).
- Rates are stored as decimals (0.035 = 3.5 %); display helpers show them
  as percentages.
- The repricing bucket boundaries are fixed at <3M / 3M-1Y / 1Y-5Y / >5Y
  because those are standard IRRBB regulatory buckets.  The finer-grained
  buckets in ``config.SIMULATION_SETTINGS`` are used by the scenario engine.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repricing bucket ordering (used for consistent display and sorting)
# ---------------------------------------------------------------------------

BUCKET_ORDER: list[str] = ["<3M", "3M-1Y", "1Y-5Y", ">5Y"]

# Bucket midpoints in years — used by risk_metrics for duration approximations
BUCKET_MIDPOINTS: dict[str, float] = {
    "<3M":   0.125,   # 1.5 months
    "3M-1Y": 0.625,   # 7.5 months
    "1Y-5Y": 3.0,     # 3 years
    ">5Y":   10.0,    # proxy for long-dated
}


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

class Instrument:
    """
    A single interest-bearing balance sheet line item.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"Fixed Rate Mortgages"``).
    side : str
        ``"asset"`` or ``"liability"``.
    notional : float
        Outstanding balance in $M.  Must be > 0.
    rate_type : str
        ``"fixed"`` — coupon does not change between repricing events.
        ``"floating"`` — coupon tracks a benchmark; reprices at
        ``repricing_tenor_months`` intervals.
    coupon : float
        Current all-in annual rate as a decimal (e.g. ``0.035`` = 3.5 %).
        For floating instruments this should be benchmark + spread at
        inception / last reset.  Must be >= 0.
    repricing_tenor_months : float
        Months between contractual repricing events.  Use 0 for
        overnight / demand instruments (bucket: ``<3M``).
        For fixed instruments this equals the contractual maturity
        in months (they reprice only at maturity).
    maturity_years : float or None
        Years to final maturity.  ``None`` for non-maturity deposits
        (demand deposits, savings).
    spread_to_benchmark : float
        For floating instruments: spread over the reference rate in
        basis points (e.g. ``150`` for SOFR+150bp).  Zero for fixed.
    behavioural_maturity_years : float or None
        Regulatory / model behavioural maturity assumed for non-maturity
        deposits (e.g. ``2.0`` years for demand deposits).  ``None``
        when not applicable.
    """

    VALID_SIDES = {"asset", "liability"}
    VALID_RATE_TYPES = {"fixed", "floating"}

    def __init__(
        self,
        name: str,
        side: str,
        notional: float,
        rate_type: str,
        coupon: float,
        repricing_tenor_months: float,
        maturity_years: Optional[float],
        spread_to_benchmark: float = 0.0,
        behavioural_maturity_years: Optional[float] = None,
    ) -> None:
        # --- Validation ---
        if not name or not name.strip():
            raise ValueError("name must be a non-empty string")
        if side not in self.VALID_SIDES:
            raise ValueError(f"side must be one of {self.VALID_SIDES!r}, got {side!r}")
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if rate_type not in self.VALID_RATE_TYPES:
            raise ValueError(f"rate_type must be one of {self.VALID_RATE_TYPES!r}, got {rate_type!r}")
        if coupon < 0:
            raise ValueError(f"coupon must be >= 0, got {coupon}")
        if repricing_tenor_months < 0:
            raise ValueError(f"repricing_tenor_months must be >= 0, got {repricing_tenor_months}")
        if maturity_years is not None and maturity_years <= 0:
            raise ValueError(f"maturity_years must be positive, got {maturity_years}")
        if behavioural_maturity_years is not None and behavioural_maturity_years <= 0:
            raise ValueError(
                f"behavioural_maturity_years must be positive, got {behavioural_maturity_years}"
            )
        if rate_type == "floating" and spread_to_benchmark == 0 and coupon > 0:
            logger.debug(
                "%s is floating with zero spread — coupon treated as absolute rate", name
            )

        self.name = name
        self.side = side
        self.notional = notional
        self.rate_type = rate_type
        self.coupon = coupon
        self.repricing_tenor_months = repricing_tenor_months
        self.maturity_years = maturity_years
        self.spread_to_benchmark = spread_to_benchmark
        self.behavioural_maturity_years = behavioural_maturity_years

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def annual_income(self) -> float:
        """
        Annual interest income (for assets) or cost (for liabilities) in $M.

        Simply ``notional × coupon``; does not account for day-count or
        amortisation.
        """
        return self.notional * self.coupon

    @property
    def repricing_bucket(self) -> str:
        """
        Standard IRRBB repricing time bucket.

        Boundaries
        ----------
        - ``<3M``   : repricing_tenor_months < 3
        - ``3M-1Y`` : 3 <= repricing_tenor_months < 12
        - ``1Y-5Y`` : 12 <= repricing_tenor_months < 60
        - ``>5Y``   : repricing_tenor_months >= 60
        """
        m = self.repricing_tenor_months
        if m < 3:
            return "<3M"
        elif m < 12:
            return "3M-1Y"
        elif m < 60:
            return "1Y-5Y"
        else:
            return ">5Y"

    @property
    def effective_maturity_years(self) -> Optional[float]:
        """
        Behavioural maturity when set, otherwise contractual maturity.

        Used by the EVE engine when discounting non-maturity deposits.
        """
        if self.behavioural_maturity_years is not None:
            return self.behavioural_maturity_years
        return self.maturity_years

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a flat dict of all fields (rates in %, notional in $M)."""
        return {
            "side":                       self.side,
            "name":                       self.name,
            "notional":                   self.notional,
            "rate_type":                  self.rate_type,
            "coupon_pct":                 round(self.coupon * 100, 3),
            "repricing_tenor_months":     self.repricing_tenor_months,
            "maturity_years":             self.maturity_years,
            "spread_to_benchmark_bp":     self.spread_to_benchmark,
            "behavioural_maturity_years": self.behavioural_maturity_years,
            "annual_income_$M":           round(self.annual_income, 3),
            "repricing_bucket":           self.repricing_bucket,
        }

    def __repr__(self) -> str:
        return (
            f"Instrument({self.name!r}, side={self.side!r}, "
            f"notional={self.notional:.0f}, coupon={self.coupon*100:.2f}%, "
            f"repricing={self.repricing_tenor_months}M, bucket={self.repricing_bucket!r})"
        )


# ---------------------------------------------------------------------------
# BalanceSheet
# ---------------------------------------------------------------------------

class BalanceSheet:
    """
    Aggregated balance sheet composed of :class:`Instrument` objects.

    The balance sheet is validated at construction time:

    - All entries in ``assets`` must have ``side == "asset"``
    - All entries in ``liabilities`` must have ``side == "liability"``
    - A warning is logged when ``|total_assets − (total_liabilities + equity)|``
      exceeds 1 % of total assets (soft check — simulation still proceeds).

    Parameters
    ----------
    assets : list[Instrument]
        Asset-side instruments.
    liabilities : list[Instrument]
        Liability-side instruments.
    equity : float
        Total equity in $M (not interest-bearing).
    """

    _BALANCE_TOLERANCE = 0.01   # 1 % relative imbalance triggers a warning

    def __init__(
        self,
        assets: list[Instrument],
        liabilities: list[Instrument],
        equity: float,
    ) -> None:
        if not assets:
            raise ValueError("assets list cannot be empty")
        if not liabilities:
            raise ValueError("liabilities list cannot be empty")
        if equity < 0:
            raise ValueError(f"equity must be non-negative, got {equity}")

        for instr in assets:
            if instr.side != "asset":
                raise ValueError(
                    f"Instrument {instr.name!r} has side={instr.side!r}; "
                    "expected 'asset' in the assets list"
                )
        for instr in liabilities:
            if instr.side != "liability":
                raise ValueError(
                    f"Instrument {instr.name!r} has side={instr.side!r}; "
                    "expected 'liability' in the liabilities list"
                )

        self.assets = list(assets)
        self.liabilities = list(liabilities)
        self.equity = equity

        self._warn_if_imbalanced()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _warn_if_imbalanced(self) -> None:
        """Log a warning when assets ≠ liabilities + equity beyond tolerance."""
        funding = self.total_liabilities + self.equity
        gap = self.total_assets - funding
        rel = abs(gap) / max(self.total_assets, 1.0)
        if rel > self._BALANCE_TOLERANCE:
            logger.warning(
                "Balance sheet is not in balance: "
                "total_assets=%.1f  total_liabilities+equity=%.1f  gap=%.1f (%.1f%%)",
                self.total_assets, funding, gap, rel * 100,
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls) -> "BalanceSheet":
        """
        Construct a ``BalanceSheet`` from ``config.BALANCE_SHEET_DEFAULTS``.

        The config stores instruments as a list of dicts; each dict must
        contain all required ``Instrument.__init__`` keyword arguments
        (everything except ``side``, which is inferred from the list key).

        Returns
        -------
        BalanceSheet
        """
        raw = config.BALANCE_SHEET_DEFAULTS

        assets = [
            Instrument(side="asset", **spec)
            for spec in raw["assets"]
        ]
        liabilities = [
            Instrument(side="liability", **spec)
            for spec in raw["liabilities"]
        ]
        equity = float(raw["equity"])

        logger.info(
            "BalanceSheet loaded from config: %d assets, %d liabilities, equity=%.1f",
            len(assets), len(liabilities), equity,
        )
        return cls(assets, liabilities, equity)

    # ------------------------------------------------------------------
    # Totals & NII
    # ------------------------------------------------------------------

    @property
    def total_assets(self) -> float:
        """Sum of all asset notionals ($M)."""
        return sum(i.notional for i in self.assets)

    @property
    def total_liabilities(self) -> float:
        """Sum of all liability notionals ($M)."""
        return sum(i.notional for i in self.liabilities)

    @property
    def net_interest_income(self) -> float:
        """
        Annualised NII at current coupon rates ($M).

        ``NII = Σ(asset_notional × asset_coupon) − Σ(liab_notional × liab_coupon)``
        """
        interest_income  = sum(i.annual_income for i in self.assets)
        interest_expense = sum(i.annual_income for i in self.liabilities)
        return interest_income - interest_expense

    @property
    def net_interest_margin(self) -> float:
        """NII ÷ total_assets, expressed as a percentage."""
        if self.total_assets == 0:
            return 0.0
        return self.net_interest_income / self.total_assets * 100

    # ------------------------------------------------------------------
    # Tabular views
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a tidy instrument-level DataFrame sorted by side and bucket.

        Columns
        -------
        side, name, notional, rate_type, coupon_pct, repricing_tenor_months,
        maturity_years, spread_to_benchmark_bp, behavioural_maturity_years,
        annual_income_$M, repricing_bucket

        Returns
        -------
        pd.DataFrame
        """
        rows = [i.to_dict() for i in self.assets + self.liabilities]
        df = pd.DataFrame(rows)

        df["side"] = pd.Categorical(
            df["side"], categories=["asset", "liability"], ordered=True
        )
        df["repricing_bucket"] = pd.Categorical(
            df["repricing_bucket"], categories=BUCKET_ORDER, ordered=True
        )
        return df.sort_values(["side", "repricing_bucket"]).reset_index(drop=True)

    def repricing_summary(self) -> pd.DataFrame:
        """
        Aggregate notionals and compute the gap ladder by repricing bucket.

        Buckets follow standard IRRBB reporting: <3M, 3M-1Y, 1Y-5Y, >5Y.

        Returns
        -------
        pd.DataFrame
            Columns: ``bucket``, ``asset_notional``, ``asset_pct``,
            ``liability_notional``, ``liability_pct``, ``gap``,
            ``cumulative_gap``.
        """
        df = self.to_dataframe()

        asset_by_bucket = (
            df[df["side"] == "asset"]
            .groupby("repricing_bucket", observed=True)["notional"]
            .sum()
            .reindex(BUCKET_ORDER, fill_value=0.0)
        )
        liab_by_bucket = (
            df[df["side"] == "liability"]
            .groupby("repricing_bucket", observed=True)["notional"]
            .sum()
            .reindex(BUCKET_ORDER, fill_value=0.0)
        )

        summary = pd.DataFrame({
            "bucket":              BUCKET_ORDER,
            "asset_notional":      asset_by_bucket.values,
            "liability_notional":  liab_by_bucket.values,
        })

        summary["gap"] = summary["asset_notional"] - summary["liability_notional"]
        summary["cumulative_gap"] = summary["gap"].cumsum()

        ta = self.total_assets  or 1.0
        tl = self.total_liabilities or 1.0
        summary["asset_pct"]     = (summary["asset_notional"]     / ta * 100).round(1)
        summary["liability_pct"] = (summary["liability_notional"] / tl * 100).round(1)

        # Reorder columns for readability
        summary = summary[[
            "bucket",
            "asset_notional", "asset_pct",
            "liability_notional", "liability_pct",
            "gap", "cumulative_gap",
        ]]
        return summary.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def display(self) -> None:
        """
        Pretty-print the full balance sheet and repricing summary to stdout.

        Layout
        ------
        1. Assets table
        2. Liabilities table
        3. Summary totals (NII, NIM, equity)
        4. Repricing gap ladder
        """
        SEP  = "=" * 100
        DASH = "-" * 100

        _FMT_COLS = [
            "name", "notional", "rate_type", "coupon_pct",
            "repricing_tenor_months", "maturity_years",
            "spread_to_benchmark_bp", "annual_income_$M", "repricing_bucket",
        ]
        _COL_LABELS = {
            "name":                    "Instrument",
            "notional":                "Notional($M)",
            "rate_type":               "Rate",
            "coupon_pct":              "Coupon(%)",
            "repricing_tenor_months":  "Repr(M)",
            "maturity_years":          "Mat(Y)",
            "spread_to_benchmark_bp":  "Sprd(bp)",
            "annual_income_$M":        "Ann.Inc($M)",
            "repricing_bucket":        "Bucket",
        }

        df = self.to_dataframe()

        def _section(side: str, label: str) -> None:
            sub = df[df["side"] == side][_FMT_COLS].copy()
            sub = sub.rename(columns=_COL_LABELS)
            # Totals row
            totals = {
                "Instrument":    "TOTAL",
                "Notional($M)":  sub["Notional($M)"].sum(),
                "Ann.Inc($M)":   sub["Ann.Inc($M)"].sum(),
            }
            totals_row = pd.DataFrame([totals])
            sub = pd.concat([sub, totals_row], ignore_index=True)
            print(f"\n  {label}")
            print(DASH)
            with pd.option_context(
                "display.float_format", "{:,.2f}".format,
                "display.max_colwidth", 32,
            ):
                print(sub.to_string(index=False))

        print(SEP)
        print("  SYNTHETIC BANK – BALANCE SHEET")
        print(SEP)

        _section("asset",     "ASSETS")
        _section("liability", "LIABILITIES")

        funding = self.total_liabilities + self.equity
        print(f"\n{DASH}")
        print(
            f"  {'Total Assets':30s}  {self.total_assets:>10.1f}  $M"
        )
        print(
            f"  {'Total Liabilities':30s}  {self.total_liabilities:>10.1f}  $M"
        )
        print(
            f"  {'Equity':30s}  {self.equity:>10.1f}  $M"
        )
        print(
            f"  {'Total Funding (Liab + Equity)':30s}  {funding:>10.1f}  $M"
        )
        print(DASH)
        print(
            f"  {'Net Interest Income (annualised)':30s}  {self.net_interest_income:>10.2f}  $M"
        )
        print(
            f"  {'Net Interest Margin':30s}  {self.net_interest_margin:>10.3f}  %"
        )

        print(f"\n{SEP}")
        print("  REPRICING GAP LADDER")
        print(DASH)
        summary = self.repricing_summary()
        with pd.option_context("display.float_format", "{:,.1f}".format):
            print(summary.to_string(index=False))
        print(SEP)
