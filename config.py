"""
config.py
---------
Central configuration for the NII/EVE simulator.

Contains:
- FRED API credentials
- Interest rate shock scenario definitions
- Default balance sheet parameters
"""

# ---------------------------------------------------------------------------
# FRED API
# ---------------------------------------------------------------------------

FRED_API_KEY: str = "6d4925644736cd94b7ab603da3b1ac2c"

# ---------------------------------------------------------------------------
# Rate shock scenarios (basis points)
# ---------------------------------------------------------------------------

RATE_SCENARIOS: dict = {
    # ------------------------------------------------------------------
    # Parallel shifts
    # ------------------------------------------------------------------
    "parallel_up_200":   {"label": "+200 bp Parallel",      "type": "parallel", "shock_bp": +200},
    "parallel_down_200": {"label": "-200 bp Parallel",      "type": "parallel", "shock_bp": -200},
    "parallel_up_100":   {"label": "+100 bp Parallel",      "type": "parallel", "shock_bp": +100},
    "parallel_down_100": {"label": "-100 bp Parallel",      "type": "parallel", "shock_bp": -100},

    # ------------------------------------------------------------------
    # Non-parallel (BCBS/EBA IRRBB standard shock profiles)
    # Short-end plateau ≤2Y; linear transition 2Y–10Y; long-end plateau ≥10Y
    # ------------------------------------------------------------------
    "bear_steepener": {
        "label": "Bear Steepener (short +200 bp, long +100 bp)",
        "type": "bear_steepener",
    },
    "bull_flattener": {
        "label": "Bull Flattener (short -100 bp, long -200 bp)",
        "type": "bull_flattener",
    },
    "bear_flattener": {
        "label": "Bear Flattener (short +200 bp, long flat)",
        "type": "bear_flattener",
    },
    "bull_steepener": {
        "label": "Bull Steepener (short flat, long -200 bp)",
        "type": "bull_steepener",
    },
}

# ---------------------------------------------------------------------------
# Default balance sheet parameters
# ---------------------------------------------------------------------------
#
# Each instrument dict must supply exactly the keyword arguments accepted by
# models.balance_sheet.Instrument.__init__ (everything except ``side``,
# which is determined by which list the entry belongs to).
#
# Field reference
# ---------------
# name                      : str   — display label
# notional                  : float — outstanding balance ($M)
# rate_type                 : str   — "fixed" | "floating"
# coupon                    : float — current all-in annual rate (decimal)
# repricing_tenor_months    : float — months between repricing events
#                                     (= maturity months for fixed instruments)
# maturity_years            : float | None — None for non-maturity deposits
# spread_to_benchmark       : float — bp over benchmark for floating (0 for fixed)
# behavioural_maturity_years: float | None — model maturity for NMDs (optional)
#
# Monetary totals
# ---------------
#   Total assets     =  300 + 200 + 150 + 100 + 50  =   800 $M
#   Total liabilities=  200 + 300 + 150 + 250        =   900 $M
#   Equity           =                                   100 $M
#   Total funding    =                                  1000 $M
#
# Note: assets (800) < funding (1000).  This synthetic balance sheet is
# intentionally simplified; the 200 $M gap is flagged as a warning at
# runtime.  Scale notionals or add instruments to close it if needed.

BALANCE_SHEET_DEFAULTS: dict = {

    # ------------------------------------------------------------------
    # Assets ($M)
    # ------------------------------------------------------------------
    "assets": [
        {
            # Long-dated, fixed-coupon mortgage book.
            # Reprices only at contractual maturity (10Y = 120 months).
            "name":                       "Fixed Rate Mortgages",
            "notional":                   300.0,
            "rate_type":                  "fixed",
            "coupon":                     0.035,   # 3.50 %
            "repricing_tenor_months":     120.0,   # = maturity
            "maturity_years":             10.0,
            "spread_to_benchmark":        0.0,
        },
        {
            # Syndicated / corporate loans priced at SOFR + 150 bp.
            # Reset every 3 months; coupon reflects SOFR 3.68 % + 1.50 %.
            "name":                       "Floating Rate Loans",
            "notional":                   200.0,
            "rate_type":                  "floating",
            "coupon":                     0.0518,  # ≈ SOFR 3.68% + 150bp
            "repricing_tenor_months":     3.0,
            "maturity_years":             5.0,
            "spread_to_benchmark":        150.0,   # bp over SOFR
        },
        {
            # Investment-grade government / agency bonds, held AFS.
            # Fixed coupon; reprices only at maturity (5Y = 60 months).
            "name":                       "Government Bonds",
            "notional":                   150.0,
            "rate_type":                  "fixed",
            "coupon":                     0.028,   # 2.80 %
            "repricing_tenor_months":     60.0,    # = maturity
            "maturity_years":             5.0,
            "spread_to_benchmark":        0.0,
        },
        {
            # Treasury bills and short CP; mature / roll within 6 months.
            # Fixed coupon for the current holding period.
            "name":                       "Short-Term Securities",
            "notional":                   100.0,
            "rate_type":                  "fixed",
            "coupon":                     0.045,   # 4.50 % (≈ 6M T-bill)
            "repricing_tenor_months":     6.0,     # = maturity
            "maturity_years":             0.5,
            "spread_to_benchmark":        0.0,
        },
        {
            # Reserve balances at the central bank; earn IORB ≈ SOFR.
            # Reprices effectively overnight (modelled as 1 month for
            # bucketing purposes).
            "name":                       "Cash & Reserves",
            "notional":                   50.0,
            "rate_type":                  "floating",
            "coupon":                     0.0368,  # ≈ SOFR 3.68%
            "repricing_tenor_months":     1.0,     # overnight → <3M bucket
            "maturity_years":             None,
            "spread_to_benchmark":        0.0,
        },
    ],

    # ------------------------------------------------------------------
    # Liabilities ($M)
    # ------------------------------------------------------------------
    "liabilities": [
        {
            # Fixed-term retail CDs / term deposits; mature in 2Y.
            "name":                       "Fixed Rate Deposits",
            "notional":                   200.0,
            "rate_type":                  "fixed",
            "coupon":                     0.015,   # 1.50 %
            "repricing_tenor_months":     24.0,    # = maturity
            "maturity_years":             2.0,
            "spread_to_benchmark":        0.0,
        },
        {
            # High-yield savings; bank reprices the rate quarterly.
            "name":                       "Savings Accounts",
            "notional":                   300.0,
            "rate_type":                  "floating",
            "coupon":                     0.025,   # 2.50 % (bank-set)
            "repricing_tenor_months":     3.0,
            "maturity_years":             None,
            "spread_to_benchmark":        0.0,
        },
        {
            # FHLB advances / senior unsecured notes; fixed coupon, 3Y term.
            "name":                       "Wholesale Funding",
            "notional":                   150.0,
            "rate_type":                  "fixed",
            "coupon":                     0.022,   # 2.20 %
            "repricing_tenor_months":     36.0,    # = maturity
            "maturity_years":             3.0,
            "spread_to_benchmark":        0.0,
        },
        {
            # Operational / transactional demand deposits.
            # Contractually repayable on demand but modelled with a
            # 2-year behavioural maturity (typical IRRBB assumption).
            "name":                       "Demand Deposits",
            "notional":                   250.0,
            "rate_type":                  "floating",
            "coupon":                     0.002,   # 0.20 %
            "repricing_tenor_months":     24.0,    # behavioural repricing
            "maturity_years":             None,
            "spread_to_benchmark":        0.0,
            "behavioural_maturity_years": 2.0,
        },
    ],

    # ------------------------------------------------------------------
    # Equity ($M) — not interest-bearing
    # ------------------------------------------------------------------
    "equity": 100.0,
}

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------

SIMULATION_SETTINGS: dict = {
    "nii_horizon_yr":       1.0,       # NII simulation horizon in years
    "eve_discount_curve":   "sofr",    # Curve used for EVE discounting
    "repricing_buckets_yr": [          # Time bucket boundaries in years
        0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float("inf")
    ],
}
