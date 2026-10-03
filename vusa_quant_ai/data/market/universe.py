"""Reference universe for cross-asset, breadth and correlation analysis."""

# key -> (yahoo symbol, description, group)
REFERENCE_ASSETS = {
    "SPX": ("^GSPC", "S&P 500 index", "index"),
    "NDX": ("^NDX", "Nasdaq 100 index", "index"),
    "DJI": ("^DJI", "Dow Jones Industrial Average", "index"),
    "RUT": ("^RUT", "Russell 2000 index", "index"),
    "RSP": ("RSP", "S&P 500 equal-weight ETF (breadth proxy)", "breadth"),
    "VIX": ("^VIX", "CBOE Volatility Index", "volatility"),
    "VIX3M": ("^VIX3M", "CBOE 3-month Volatility Index", "volatility"),
    "TNX": ("^TNX", "US 10Y Treasury yield (x10)", "rates"),
    "IRX": ("^IRX", "US 13-week T-bill yield (x10)", "rates"),
    "TLT": ("TLT", "20+Y Treasury ETF", "rates"),
    "HYG": ("HYG", "High-yield credit ETF", "credit"),
    "LQD": ("LQD", "Investment-grade credit ETF", "credit"),
    "DXY": ("DX-Y.NYB", "US Dollar index", "fx"),
    "EURUSD": ("EURUSD=X", "EUR/USD", "fx"),
    "GOLD": ("GC=F", "Gold futures", "commodity"),
    "OIL": ("CL=F", "WTI crude futures", "commodity"),
}

SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLV": "Health Care", "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples", "XLE": "Energy", "XLI": "Industrials", "XLB": "Materials",
    "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
}

# Expected sign of the relationship used only for synthetic demo generation (betas)
SYNTHETIC_BETAS = {"SPX": 1.0, "NDX": 1.25, "DJI": 0.9, "RUT": 1.2, "RSP": 0.95, "VIX": -4.0, "VIX3M": -2.5,
                   "TNX": 0.3, "IRX": 0.05, "TLT": -0.2, "HYG": 0.35, "LQD": 0.1, "DXY": -0.1, "EURUSD": 0.1,
                   "GOLD": 0.05, "OIL": 0.5}
SYNTHETIC_BASES = {"VIX": 18.0, "VIX3M": 20.0, "TNX": 40.0, "IRX": 45.0, "EURUSD": 1.1, "DXY": 100.0}
