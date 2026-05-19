from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

try:
	from pmdarima.arima import auto_arima
except Exception:
	auto_arima = None

try:
	from arch import arch_model
except Exception:
	arch_model = None


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
	page_title="Ticker Forecast Lab",
	page_icon="📈",
	layout="wide",
	initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Fonts ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background ─────────────────────────────────────────── */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* ── Main content area ──────────────────────────────────────── */
section.main > div {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* ── Hero title ─────────────────────────────────────────────── */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    font-weight: 400;
    color: #f0f2f8;
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin: 0 0 0.25rem 0;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #7b8299;
    font-weight: 300;
    margin: 0 0 1.8rem 0;
    letter-spacing: 0.2px;
}

/* ── Divider ────────────────────────────────────────────────── */
.section-divider {
    height: 1px;
    background: linear-gradient(to right, #2a2d3a, #3d4255 40%, #2a2d3a);
    margin: 1.8rem 0;
    border: none;
}

/* ── Sidebar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #11141c;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Serif Display', serif;
    color: #c9ccda;
    font-size: 1rem;
    font-weight: 400;
    margin: 1.1rem 0 0.4rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #1e2130;
}
[data-testid="stSidebar"] label {
    font-size: 0.82rem;
    color: #8a8fa8;
    letter-spacing: 0.4px;
    font-weight: 500;
}
[data-testid="stSidebar"] .stSlider > label {
    color: #9da3bc;
}

/* ── Run button ─────────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4f6ef7 0%, #7b52f7 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
    letter-spacing: 0.5px;
    padding: 0.65rem 1rem;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
    box-shadow: 0 4px 18px rgba(79,110,247,0.35);
    margin-top: 0.5rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}

/* ── Metric cards ───────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #161923;
    border: 1px solid #242840;
    border-radius: 10px;
    padding: 1rem 1.2rem 0.8rem;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: #4f6ef7;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    color: #6e7490 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.8rem !important;
    color: #e8eaf0 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
}

/* ── DataFrames ─────────────────────────────────────────────── */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #1e2130 !important;
}

/* ── Tabs ───────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.2rem;
    background: #11141c;
    border-radius: 10px;
    padding: 0.3rem;
    border: 1px solid #1e2130;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.86rem;
    font-weight: 500;
    color: #7b8299;
    padding: 0.4rem 1rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f6ef7 0%, #7b52f7 100%) !important;
    color: #fff !important;
}

/* ── Info / warning / error boxes ──────────────────────────── */
.stAlert {
    border-radius: 8px;
    border-left-width: 4px;
}

/* ── Download button ────────────────────────────────────────── */
.stDownloadButton > button {
    background: transparent;
    border: 1px solid #2e3450;
    color: #9da3bc;
    border-radius: 7px;
    font-size: 0.83rem;
    padding: 0.38rem 0.85rem;
    transition: all 0.2s;
}
.stDownloadButton > button:hover {
    border-color: #4f6ef7;
    color: #c6cbe8;
    background: rgba(79,110,247,0.08);
}

/* ── Selectbox / Radio ──────────────────────────────────────── */
.stSelectbox label, .stRadio label {
    font-size: 0.83rem;
    color: #8a8fa8;
}

/* ── Section headers inside main ───────────────────────────── */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #d5d9eb;
    font-weight: 400;
    margin: 0.2rem 0 0.8rem 0;
    letter-spacing: -0.2px;
}

/* ── Callout card ───────────────────────────────────────────── */
.callout-card {
    background: #161923;
    border: 1px solid #242840;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}

/* ── Empty state ────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #444867;
}
.empty-state .icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    display: block;
}
.empty-state h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #6e7490;
    font-weight: 400;
    margin-bottom: 0.5rem;
}
.empty-state p {
    font-size: 0.9rem;
    color: #444867;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Monospace numbers in tables ────────────────────────────── */
[data-testid="stDataFrame"] td {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}

/* ── Caption text ────────────────────────────────────────────── */
.stCaption {
    color: #5a5f77 !important;
    font-size: 0.78rem !important;
}

/* ── Tooltip/help text ───────────────────────────────────────── */
.stTooltipIcon {
    color: #4f6ef7 !important;
}

/* ── Ticker badge ────────────────────────────────────────────── */
.ticker-badge {
    display: inline-block;
    background: rgba(79,110,247,0.12);
    border: 1px solid rgba(79,110,247,0.3);
    color: #8fa5fb;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    letter-spacing: 0.5px;
    margin-left: 0.4rem;
    vertical-align: middle;
}

/* ── Model pill ──────────────────────────────────────────────── */
.model-count-badge {
    display: inline-block;
    background: rgba(123,82,247,0.15);
    border: 1px solid rgba(123,82,247,0.3);
    color: #b09afa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.1rem 0.55rem;
    border-radius: 20px;
    margin-left: 0.35rem;
    vertical-align: middle;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4f6ef7; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME  (dark, consistent with the UI)
# ─────────────────────────────────────────────────────────────────────────────
def apply_plot_theme() -> None:
	plt.rcParams.update({
		"figure.facecolor":  "#161923",
		"axes.facecolor":    "#161923",
		"axes.edgecolor":    "#2a2d3a",
		"axes.labelcolor":   "#9da3bc",
		"axes.titlecolor":   "#d5d9eb",
		"axes.titlesize":    11,
		"axes.titleweight":  "bold",
		"axes.titlepad":     10,
		"axes.grid":         True,
		"axes.spines.top":   False,
		"axes.spines.right": False,
		"grid.color":        "#1e2130",
		"grid.linewidth":    0.7,
		"xtick.color":       "#6e7490",
		"ytick.color":       "#6e7490",
		"xtick.labelsize":   8,
		"ytick.labelsize":   8,
		"legend.facecolor":  "#1a1e2a",
		"legend.edgecolor":  "#2a2d3a",
		"legend.fontsize":   8.5,
		"legend.framealpha": 0.9,
		"lines.linewidth":   1.6,
		"text.color":        "#c4c9de",
		"font.family":       "sans-serif",
		"font.size":         9,
		"figure.dpi":        130,
	})

apply_plot_theme()

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TICKER_OPTIONS: Dict[str, str] = {
	"NIFTY 50 (^NSEI)": "^NSEI",
	"NIFTY Bank (^NSEBANK)": "^NSEBANK",
	"BSE Sensex (^BSESN)": "^BSESN",
	"Reliance (RELIANCE.NS)": "RELIANCE.NS",
	"TCS (TCS.NS)": "TCS.NS",
	"Infosys (INFY.NS)": "INFY.NS",
	"HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
	"ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
	"SBI (SBIN.NS)": "SBIN.NS",
	"ITC (ITC.NS)": "ITC.NS",
}

# Accessible, distinct colour palette for forecasts
FORECAST_PALETTE = [
	"#4f8ef7",   # blue
	"#f7634f",   # coral
	"#50d89a",   # mint
	"#f7c34f",   # amber
	"#c44ff7",   # violet
	"#4ff7e8",   # cyan
	"#f74fb8",   # pink
	"#b0d94f",   # lime
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES & HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ForecastResult:
	name: str
	forecast: pd.Series
	metrics: Dict[str, float]
	model_info: str
	residuals: Optional[pd.Series] = None
	aux_series: Optional[pd.Series] = None


def safe_mape(actual: pd.Series, predicted: pd.Series) -> float:
	actual_aligned, predicted_aligned = actual.align(predicted, join="inner")
	non_zero = actual_aligned != 0
	if not non_zero.any():
		return float("nan")
	return float(
		np.mean(
			np.abs(
				(actual_aligned[non_zero] - predicted_aligned[non_zero])
				/ actual_aligned[non_zero]
			)
		)
		* 100
	)


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
	actual_aligned, predicted_aligned = actual.align(predicted, join="inner")
	return {
		"MAE":  float(mean_absolute_error(actual_aligned, predicted_aligned)),
		"RMSE": float(np.sqrt(mean_squared_error(actual_aligned, predicted_aligned))),
		"MAPE": safe_mape(actual_aligned, predicted_aligned),
		"R²":   float(r2_score(actual_aligned, predicted_aligned)),
	}


def interpret_p_value(
	p_value: float,
	alpha: float,
	null_hypothesis: str,
	reject_message: str,
	fail_message: str,
) -> str:
	if p_value < alpha:
		return f"p={p_value:.4f} < {alpha} → Reject H₀ ({null_hypothesis}). {reject_message}"
	return f"p={p_value:.4f} ≥ {alpha} → Fail to reject H₀ ({null_hypothesis}). {fail_message}"


def format_ticker_details(details: Dict[str, object]) -> pd.DataFrame:
	rows = []
	for label, key in [
		("Symbol", "symbol"), ("Name", "name"), ("Exchange", "exchange"),
		("Type", "quoteType"), ("Currency", "currency"), ("Sector", "sector"),
		("Industry", "industry"), ("Market Cap", "marketCap"),
		("Last Price", "regularMarketPrice"), ("Prev Close", "previousClose"),
		("Open", "open"), ("Day High", "dayHigh"), ("Day Low", "dayLow"),
		("52W Low", "fiftyTwoWeekLow"), ("52W High", "fiftyTwoWeekHigh"),
	]:
		value = details.get(key)
		if value is None or value == "":
			continue
		if isinstance(value, (int, float, np.integer, np.floating)) and key in {
			"marketCap", "regularMarketPrice", "previousClose", "open",
			"dayHigh", "dayLow", "fiftyTwoWeekLow", "fiftyTwoWeekHigh",
		}:
			value = f"{value:,.0f}" if key == "marketCap" else f"{float(value):,.2f}"
		rows.append({"Field": label, "Value": value})
	return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CACHING
# ─────────────────────────────────────────────────────────────────────────────
def get_rolling_checkpoint_cache() -> Dict[str, ForecastResult]:
	if "rolling_forecast_cache" not in st.session_state:
		st.session_state["rolling_forecast_cache"] = {}
	return st.session_state["rolling_forecast_cache"]


def get_ticker_data_checkpoint_cache() -> Dict[str, pd.DataFrame]:
	if "ticker_data_cache" not in st.session_state:
		st.session_state["ticker_data_cache"] = {}
	return st.session_state["ticker_data_cache"]


def build_ticker_data_cache_key(ticker: str, start: str, end: str) -> str:
	return f"{ticker.strip()}|{start}|{end}"


def series_fingerprint(series: pd.Series) -> str:
	hashed = pd.util.hash_pandas_object(series, index=True).to_numpy(dtype=np.uint64).tobytes()
	return hashlib.sha256(hashed).hexdigest()


def build_rolling_cache_key(
	model_name: str,
	train_prices: pd.Series,
	test_prices: pd.Series,
	params: Dict[str, object],
) -> str:
	payload = {
		"model": model_name,
		"train": series_fingerprint(train_prices),
		"test":  series_fingerprint(test_prices),
		"params": {k: v for k, v in params.items() if k != "run"},
	}
	return hashlib.sha256(
		json.dumps(payload, sort_keys=True, default=str).encode()
	).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_ticker_details(ticker: str) -> Dict[str, object]:
	ticker = ticker.strip()
	if not ticker:
		raise ValueError("Please enter a ticker symbol.")
	ticker_obj = yf.Ticker(ticker)
	history = ticker_obj.history(period="5d", auto_adjust=True)
	if history is None or history.empty:
		raise ValueError(f"'{ticker}' doesn't appear to be a valid yfinance ticker.")
	info: Dict[str, object] = {}
	try:
		raw = ticker_obj.info or {}
		if isinstance(raw, dict):
			info = raw
	except Exception:
		info = {}
	latest = history.iloc[-1]
	return {
		"symbol":             info.get("symbol", ticker),
		"name":               info.get("longName") or info.get("shortName") or ticker,
		"exchange":           info.get("exchange") or info.get("fullExchangeName"),
		"quoteType":          info.get("quoteType"),
		"currency":           info.get("currency"),
		"sector":             info.get("sector"),
		"industry":           info.get("industry"),
		"marketCap":          info.get("marketCap"),
		"regularMarketPrice": info.get("regularMarketPrice", float(latest.get("Close", np.nan))),
		"previousClose":      info.get("previousClose", float(latest.get("Close", np.nan))),
		"open":               info.get("open", float(latest.get("Open", latest.get("Close", np.nan)))),
		"dayHigh":            info.get("dayHigh", float(latest.get("High", latest.get("Close", np.nan)))),
		"dayLow":             info.get("dayLow", float(latest.get("Low", latest.get("Close", np.nan)))),
		"fiftyTwoWeekLow":    info.get("fiftyTwoWeekLow"),
		"fiftyTwoWeekHigh":   info.get("fiftyTwoWeekHigh"),
	}


@st.cache_data(show_spinner=False)
def download_ticker_data(ticker: str, start: str, end: str) -> pd.DataFrame:
	data = yf.download(
		tickers=ticker, start=start, end=end,
		interval="1d", auto_adjust=True, progress=False, threads=True,
	)
	if data is None or data.empty:
		raise ValueError(f"No data returned for {ticker}")

	frame = data.copy()
	if isinstance(frame.columns, pd.MultiIndex):
		if "Close" not in frame.columns.get_level_values(0):
			raise ValueError("Downloaded data does not contain a Close column.")
		close_frame = frame.xs("Close", axis=1, level=0)
		close_series = (
			close_frame[ticker]
			if isinstance(close_frame, pd.DataFrame) and ticker in close_frame.columns
			else close_frame.iloc[:, 0] if isinstance(close_frame, pd.DataFrame)
			else close_frame
		)
	else:
		if "Close" not in frame.columns:
			cands = [c for c in frame.columns if str(c).lower() == "close"]
			if not cands:
				raise ValueError("Downloaded data does not contain a Close column.")
			frame = frame.rename(columns={cands[0]: "Close"})
		close_series = frame["Close"]

	frame = pd.DataFrame({"price": close_series})
	return frame.sort_index().asfreq("B").ffill().dropna()


def add_log_returns(frame: pd.DataFrame) -> pd.DataFrame:
	result = frame.copy()
	result["returns"] = np.log(result["price"] / result["price"].shift(1))
	return result


def train_test_split_frame(
	data: pd.DataFrame, split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	idx = max(1, min(len(data) - 1, int(len(data) * split_ratio)))
	return data.iloc[:idx].copy(), data.iloc[idx:].copy()


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def series_constant(value: float, index: pd.Index, name: str) -> pd.Series:
	return pd.Series(np.repeat(value, len(index)), index=index, name=name)


def mean_forecast(train: pd.Series, idx: pd.Index) -> pd.Series:
	return series_constant(train.mean(), idx, "mean_forecast")


def naive_forecast(train: pd.Series, idx: pd.Index) -> pd.Series:
	return series_constant(float(train.iloc[-1]), idx, "naive_forecast")


def moving_average_forecast(train: pd.Series, idx: pd.Index, window: int) -> pd.Series:
	w = max(1, min(window, len(train)))
	return series_constant(float(train.tail(w).mean()), idx, f"ma_{w}")


def order_str(order: Tuple) -> str:
	return "(" + ", ".join(str(x) for x in order) + ")"


def price_path_from_returns(
	train: pd.Series, returns: Iterable[float], index: pd.Index
) -> pd.Series:
	price = float(train.iloc[-1])
	vals: List[float] = []
	for r in returns:
		price *= float(np.exp(r))
		vals.append(price)
	return pd.Series(vals, index=index)


def rolling_price_forecast(
	train: pd.Series, test: pd.Series, fit_fn, label: str, block: int
) -> Tuple[pd.Series, pd.Series, str]:
	history = list(train.dropna())
	vals, idx = [], []
	block = max(1, min(int(block), len(test)))
	for start in range(0, len(test), block):
		blk = test.iloc[start : start + block]
		fitted = fit_fn(pd.Series(history, index=pd.RangeIndex(len(history))))
		vals.extend(pd.Series(fitted.forecast(steps=len(blk))).tolist())
		idx.extend(blk.index.tolist())
		history.extend(blk.tolist())
	fcast = pd.Series(vals, index=pd.Index(idx), name=f"{label}_forecast")
	return fcast, (test - fcast).rename(f"{label}_residuals"), f"Rolling ({block}-day blocks)"


def rolling_return_forecast(
	train: pd.Series, test: pd.Series, fit_fn, label: str, block: int
) -> Tuple[pd.Series, pd.Series, str]:
	history = list(train.dropna())
	vals, idx = [], []
	block = max(1, min(int(block), len(test)))
	for start in range(0, len(test), block):
		blk = test.iloc[start : start + block]
		hs = pd.Series(history)
		fitted = fit_fn(hs)
		blk_fcast = price_path_from_returns(hs, fitted.forecast(steps=len(blk)), blk.index)
		vals.extend(blk_fcast.tolist())
		idx.extend(blk.index.tolist())
		history.extend(blk.tolist())
	fcast = pd.Series(vals, index=pd.Index(idx), name=f"{label}_forecast")
	return fcast, (test - fcast).rename(f"{label}_residuals"), f"Rolling ({block}-day blocks)"


def forecast_return_model(
	train: pd.Series, test: pd.Series, order: Tuple, name: str
) -> Tuple[pd.Series, str, pd.Series]:
	returns = np.log(train).diff().dropna()
	fitted = ARIMA(returns, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
	fcast = price_path_from_returns(train, fitted.forecast(steps=len(test)), test.index)
	return fcast, f"ARIMA{order_str(order)} on log returns", (test - fcast).rename(f"{name}_residuals")


def forecast_arima_prices(
	train: pd.Series, test: pd.Series, order: Tuple
) -> Tuple[pd.Series, str, pd.Series]:
	fitted = ARIMA(train, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
	fcast = pd.Series(fitted.forecast(steps=len(test)), index=test.index)
	return fcast, f"ARIMA{order_str(order)} on prices", (test - fcast).rename(f"ARIMA_residuals")


def forecast_sarima_prices(
	train: pd.Series, test: pd.Series, order: Tuple, seasonal_order: Tuple
) -> Tuple[pd.Series, str, pd.Series]:
	fitted = SARIMAX(
		train, order=order, seasonal_order=seasonal_order,
		enforce_stationarity=False, enforce_invertibility=False,
	).fit(disp=False)
	fcast = pd.Series(fitted.forecast(steps=len(test)), index=test.index)
	info = f"SARIMA{order_str(order)} × {order_str(seasonal_order)}"
	return fcast, info, (test - fcast).rename("SARIMA_residuals")


def forecast_auto_arima_prices(
	train: pd.Series, test: pd.Series, seasonal: bool, m: int
) -> Tuple[pd.Series, str, pd.Series]:
	if auto_arima is None:
		raise RuntimeError("pmdarima is not installed.")
	fitted = auto_arima(
		train, seasonal=seasonal, m=max(1, m), stepwise=True,
		suppress_warnings=True, error_action="ignore",
		information_criterion="aic", max_p=6, max_q=6, max_d=2,
	)
	fcast = pd.Series(fitted.predict(n_periods=len(test)), index=test.index)
	return fcast, f"Auto ARIMA{fitted.order}", (test - fcast).rename("AutoARIMA_residuals")


def forecast_garch_volatility(
	train: pd.Series, test_idx: pd.Index, p: int, q: int
) -> Tuple[pd.Series, str, pd.Series, pd.Series]:
	if arch_model is None:
		raise RuntimeError("arch is not installed.")
	returns = 100 * np.log(train / train.shift(1)).dropna()
	fitted = arch_model(returns, vol="GARCH", p=p, q=q, mean="Zero", dist="normal").fit(disp="off")
	fcast_prices = series_constant(float(train.iloc[-1]), test_idx, "garch_vol_proxy")
	residuals = pd.Series(fitted.resid, index=returns.index).dropna()
	in_vol = pd.Series(fitted.conditional_volatility, index=returns.index).dropna()
	forecast_obj = fitted.forecast(horizon=len(test_idx), reindex=False)
	out_vol = pd.Series(np.sqrt(forecast_obj.variance.values[-1]), index=test_idx, name="forecast_volatility")
	vol_path = pd.concat([in_vol.rename("in_sample_volatility"), out_vol])
	return fcast_prices, f"GARCH({p},{q}) volatility proxy", residuals, vol_path


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def render_adf_summary(price: pd.Series, returns: pd.Series) -> pd.DataFrame:
	rows = []
	for label, series, stat_label, statio_msg, nonstatio_msg in [
		("Price", price, "Price series", "Stationary ✓", "Non-stationary ✗"),
		("Log Returns", returns, "Log returns", "Stationary ✓", "Non-stationary ✗"),
	]:
		stat, pval, *_ = adfuller(series.dropna())
		rows.append({
			"Series":        label,
			"ADF Statistic": f"{stat:.4f}",
			"p-value":       f"{pval:.4f}",
			"Result":        statio_msg if pval < 0.05 else nonstatio_msg,
			"Interpretation": interpret_p_value(
				pval, 0.05,
				"series has a unit root",
				"Series is stationary.",
				"Series is NOT stationary.",
			),
		})
	return pd.DataFrame(rows)


def render_ljung_box_summary(residuals: pd.Series) -> pd.DataFrame:
	clean = residuals.dropna()
	if len(clean) < 10:
		return pd.DataFrame([{
			"Test": "Ljung-Box", "p-value": "N/A",
			"Interpretation": "Not enough residual observations.",
		}])
	result = acorr_ljungbox(clean, lags=[10], return_df=True)
	pval = float(result["lb_pvalue"].iloc[-1])
	return pd.DataFrame([{
		"Test": "Ljung-Box (lag 10)",
		"p-value": f"{pval:.4f}",
		"Result": "Autocorrelation present ✗" if pval < 0.05 else "White noise ✓",
		"Interpretation": interpret_p_value(
			pval, 0.05,
			"residuals are independent",
			"Residuals still have autocorrelation. Model hasn't captured all structure.",
			"Residuals look like white noise — model is well-specified.",
		),
	}])


# ─────────────────────────────────────────────────────────────────────────────
# AVAILABLE MODELS
# ─────────────────────────────────────────────────────────────────────────────
def available_models() -> List[str]:
	models = ["Mean", "Naive", "Moving Average", "AR", "MA", "ARMA", "ARIMA", "SARIMA"]
	if auto_arima is not None:
		models.append("Auto ARIMA")
	if arch_model is not None:
		models.append("GARCH Volatility")
	return models


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def get_sidebar_params() -> Dict[str, object]:
	with st.sidebar:
		st.markdown(
			"<div style='padding:0.6rem 0 1rem 0;'>"
			"<span style='font-family:DM Serif Display,serif;font-size:1.15rem;color:#c9ccda;'>⚙️ Controls</span>"
			"</div>",
			unsafe_allow_html=True,
		)

		# ── Run button at the TOP for visibility ──
		run = st.button("▶ Run Forecast", use_container_width=True)
		st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

		st.markdown("### 📌 Ticker")
		ticker_source = st.radio(
			"Source", ["Preset", "Custom"],
			horizontal=True, label_visibility="collapsed",
		)
		if ticker_source == "Preset":
			ticker_label = st.selectbox("Ticker", list(TICKER_OPTIONS.keys()), index=0, label_visibility="collapsed")
			ticker = TICKER_OPTIONS[ticker_label]
		else:
			ticker = st.text_input("Ticker symbol", placeholder="e.g. AAPL, MSFT, RELIANCE.NS").strip()
			ticker_label = ticker or "Custom"

		st.markdown("### 📅 Date Range")
		_today = pd.Timestamp.today().normalize()
		_preset_map = {
			"1 Year":   _today - pd.DateOffset(years=1),
			"5 Years":  _today - pd.DateOffset(years=5),
			"10 Years": _today - pd.DateOffset(years=10),
			"Custom":   None,
		}
		preset_choice = st.radio(
			"Period",
			list(_preset_map.keys()),
			index=1,
			horizontal=True,
			label_visibility="collapsed",
		)
		if preset_choice == "Custom":
			col_a, col_b = st.columns(2)
			with col_a:
				start_date = st.date_input("From", value=(_today - pd.DateOffset(years=5)).date())
			with col_b:
				end_date = st.date_input("To", value=_today.date())
		else:
			start_date = _preset_map[preset_choice].date()
			end_date   = _today.date()
			st.caption(f"📆 {start_date}  →  {end_date}")

		st.markdown("### 🔧 Data Settings")
		split_ratio = st.slider("Train split", 0.50, 0.95, 0.80, 0.01,
								help="Fraction of data used for training.")
		horizon = st.slider("Forecast horizon (days)", 5, 252, 30,
							help="Number of future days to forecast beyond the test set.")
		recent_rows = st.slider("Max rows to use", 250, 4000, 2500, 50,
								help="Caps the series length — reduces fitting time for slow models.")
		forecast_mode = st.radio(
			"Forecast mode",
			["Entire test at once", "Rolling forecast"],
			help="Rolling re-fits the model on each block of actuals for a more realistic evaluation.",
		)
		rolling_window = st.select_slider(
			"Rolling block size (days)",
			options=[5, 10, 20], value=10,
			disabled=forecast_mode != "Rolling forecast",
		)

		st.markdown("### 🤖 Models")
		selected_models = st.multiselect(
			"Select models",
			available_models(),
			default=["Mean", "Naive", "Moving Average", "ARIMA"],
			label_visibility="collapsed",
		)

		st.markdown("### 📊 Model Parameters")

		with st.expander("Baseline — Moving Average"):
			ma_window = st.slider("Window", 3, 252, 30)

		with st.expander("AR / MA / ARMA"):
			ar_p   = st.slider("AR order p",   1, 12, 1)
			ma_q   = st.slider("MA order q",   1, 12, 1)
			arma_p = st.slider("ARMA p",       1, 12, 1)
			arma_q = st.slider("ARMA q",       1, 12, 1)

		with st.expander("ARIMA"):
			arima_p = st.slider("ARIMA p", 0, 12, 1)
			arima_d = st.slider("ARIMA d", 0,  2, 1)
			arima_q = st.slider("ARIMA q", 0, 12, 1)

		with st.expander("SARIMA"):
			sarima_p = st.slider("p", 0, 6, 1)
			sarima_d = st.slider("d", 0, 2, 1)
			sarima_q = st.slider("q", 0, 6, 1)
			sarima_P = st.slider("Seasonal P", 0, 4, 1)
			sarima_D = st.slider("Seasonal D", 0, 2, 1)
			sarima_Q = st.slider("Seasonal Q", 0, 4, 1)
			seasonal_period = st.slider("Season length m", 2, 31, 5)

		with st.expander("Optional — Auto ARIMA / GARCH"):
			use_auto_arima    = st.checkbox("Enable Auto ARIMA",        value=False, disabled=auto_arima is None)
			auto_arima_seasonal = st.checkbox("Auto ARIMA — seasonal",  value=False, disabled=auto_arima is None)
			garch_p   = st.slider("GARCH p", 1, 5, 1)
			garch_q   = st.slider("GARCH q", 1, 5, 1)
			use_garch = st.checkbox("Enable GARCH volatility proxy", value=False, disabled=arch_model is None)

		st.markdown("---")
		st.caption("Last forecast stays visible until you run a new one.")

	return {
		"ticker": ticker, "ticker_label": ticker_label, "ticker_source": ticker_source,
		"start_date": str(start_date), "end_date": str(end_date),
		"split_ratio": split_ratio, "horizon": horizon, "recent_rows": recent_rows,
		"forecast_mode": forecast_mode, "rolling_window": rolling_window,
		"selected_models": selected_models, "ma_window": ma_window,
		"ar_p": ar_p, "ma_q": ma_q, "arma_p": arma_p, "arma_q": arma_q,
		"arima_p": arima_p, "arima_d": arima_d, "arima_q": arima_q,
		"sarima_p": sarima_p, "sarima_d": sarima_d, "sarima_q": sarima_q,
		"sarima_P": sarima_P, "sarima_D": sarima_D, "sarima_Q": sarima_Q,
		"seasonal_period": seasonal_period,
		"use_auto_arima": use_auto_arima, "auto_arima_seasonal": auto_arima_seasonal,
		"garch_p": garch_p, "garch_q": garch_q, "use_garch": use_garch,
		"run": run,
	}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_models(
	train: pd.DataFrame, test: pd.DataFrame, params: Dict[str, object]
) -> Dict[str, ForecastResult]:
	selected      = set(params["selected_models"])
	if params["use_auto_arima"]:
		selected.add("Auto ARIMA")
	if params["use_garch"]:
		selected.add("GARCH Volatility")
	use_rolling   = params["forecast_mode"] == "Rolling forecast"
	block         = int(params["rolling_window"])
	results: Dict[str, ForecastResult] = {}
	rolling_cache = get_rolling_checkpoint_cache()

	def cache_key(name: str) -> str:
		return build_rolling_cache_key(name, train["price"], test["price"], params)

	def get_cached(name: str) -> Optional[ForecastResult]:
		return rolling_cache.get(cache_key(name))

	def store(name: str, r: ForecastResult) -> ForecastResult:
		rolling_cache[cache_key(name)] = r
		return r

	# ── Baseline models ──────────────────────────────────────────────────────
	if "Mean" in selected:
		f = mean_forecast(train["price"], test.index)
		results["Mean"] = ForecastResult("Mean", f, compute_metrics(test["price"], f),
										 "Constant mean of training prices",
										 residuals=test["price"] - f)

	if "Naive" in selected:
		f = naive_forecast(train["price"], test.index)
		results["Naive"] = ForecastResult("Naive", f, compute_metrics(test["price"], f),
										  "Last observed price repeated forward",
										  residuals=test["price"] - f)

	if "Moving Average" in selected:
		w = int(params["ma_window"])
		f = moving_average_forecast(train["price"], test.index, w)
		results["Moving Average"] = ForecastResult(
			"Moving Average", f, compute_metrics(test["price"], f),
			f"Average of last {w} prices repeated forward",
			residuals=test["price"] - f)

	# ── AR ───────────────────────────────────────────────────────────────────
	if "AR" in selected:
		if use_rolling:
			cached = get_cached("AR")
			if cached is None:
				p = int(params["ar_p"])
				f, res, info = rolling_return_forecast(
					train["price"], test["price"],
					lambda s: ARIMA(pd.Series(np.log(s).diff().dropna()),
									order=(p, 0, 0),
									enforce_stationarity=False, enforce_invertibility=False).fit(),
					"AR", block,
				)
				cached = store("AR", ForecastResult(f"AR({p})", f, compute_metrics(test["price"], f),
													 f"AR({p}) on log returns — {info}", residuals=res))
			results["AR"] = cached
		else:
			p = int(params["ar_p"])
			f, info, res = forecast_return_model(train["price"], test["price"], (p, 0, 0), "AR")
			results["AR"] = ForecastResult(f"AR({p})", f, compute_metrics(test["price"], f), info, residuals=res)

	# ── MA ───────────────────────────────────────────────────────────────────
	if "MA" in selected:
		if use_rolling:
			cached = get_cached("MA")
			if cached is None:
				q = int(params["ma_q"])
				f, res, info = rolling_return_forecast(
					train["price"], test["price"],
					lambda s: ARIMA(pd.Series(np.log(s).diff().dropna()),
									order=(0, 0, q),
									enforce_stationarity=False, enforce_invertibility=False).fit(),
					"MA", block,
				)
				cached = store("MA", ForecastResult(f"MA({q})", f, compute_metrics(test["price"], f),
													 f"MA({q}) on log returns — {info}", residuals=res))
			results["MA"] = cached
		else:
			q = int(params["ma_q"])
			f, info, res = forecast_return_model(train["price"], test["price"], (0, 0, q), "MA")
			results["MA"] = ForecastResult(f"MA({q})", f, compute_metrics(test["price"], f), info, residuals=res)

	# ── ARMA ─────────────────────────────────────────────────────────────────
	if "ARMA" in selected:
		if use_rolling:
			cached = get_cached("ARMA")
			if cached is None:
				ap, aq = int(params["arma_p"]), int(params["arma_q"])
				f, res, info = rolling_return_forecast(
					train["price"], test["price"],
					lambda s: ARIMA(pd.Series(np.log(s).diff().dropna()),
									order=(ap, 0, aq),
									enforce_stationarity=False, enforce_invertibility=False).fit(),
					"ARMA", block,
				)
				cached = store("ARMA", ForecastResult(f"ARMA({ap},{aq})", f, compute_metrics(test["price"], f),
													   f"ARMA({ap},{aq}) on log returns — {info}", residuals=res))
			results["ARMA"] = cached
		else:
			ap, aq = int(params["arma_p"]), int(params["arma_q"])
			f, info, res = forecast_return_model(train["price"], test["price"], (ap, 0, aq), "ARMA")
			results["ARMA"] = ForecastResult(f"ARMA({ap},{aq})", f, compute_metrics(test["price"], f), info, residuals=res)

	# ── ARIMA ────────────────────────────────────────────────────────────────
	if "ARIMA" in selected:
		if use_rolling:
			cached = get_cached("ARIMA")
			if cached is None:
				ip, id_, iq = int(params["arima_p"]), int(params["arima_d"]), int(params["arima_q"])
				f, res, info = rolling_price_forecast(
					train["price"], test["price"],
					lambda s: ARIMA(s, order=(ip, id_, iq),
									enforce_stationarity=False, enforce_invertibility=False).fit(),
					"ARIMA", block,
				)
				cached = store("ARIMA", ForecastResult(f"ARIMA({ip},{id_},{iq})", f, compute_metrics(test["price"], f),
													   f"ARIMA({ip},{id_},{iq}) on prices — {info}", residuals=res))
			results["ARIMA"] = cached
		else:
			ip, id_, iq = int(params["arima_p"]), int(params["arima_d"]), int(params["arima_q"])
			f, info, res = forecast_arima_prices(train["price"], test["price"], (ip, id_, iq))
			results["ARIMA"] = ForecastResult(f"ARIMA({ip},{id_},{iq})", f, compute_metrics(test["price"], f), info, residuals=res)

	# ── SARIMA ───────────────────────────────────────────────────────────────
	if "SARIMA" in selected:
		if use_rolling:
			cached = get_cached("SARIMA")
			if cached is None:
				sp, sd, sq = int(params["sarima_p"]), int(params["sarima_d"]), int(params["sarima_q"])
				sP, sD, sQ, sm = int(params["sarima_P"]), int(params["sarima_D"]), int(params["sarima_Q"]), int(params["seasonal_period"])
				f, res, info = rolling_price_forecast(
					train["price"], test["price"],
					lambda s: SARIMAX(s, order=(sp, sd, sq), seasonal_order=(sP, sD, sQ, sm),
									  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False),
					"SARIMA", block,
				)
				_sname = f"SARIMA({sp},{sd},{sq})({sP},{sD},{sQ},{sm})"
				cached = store("SARIMA", ForecastResult(_sname, f, compute_metrics(test["price"], f),
														 f"{_sname} — {info}", residuals=res))
			results["SARIMA"] = cached
		else:
			sp, sd, sq = int(params["sarima_p"]), int(params["sarima_d"]), int(params["sarima_q"])
			sP, sD, sQ, sm = int(params["sarima_P"]), int(params["sarima_D"]), int(params["sarima_Q"]), int(params["seasonal_period"])
			f, info, res = forecast_sarima_prices(
				train["price"], test["price"], (sp, sd, sq), (sP, sD, sQ, sm),
			)
			_sname = f"SARIMA({sp},{sd},{sq})({sP},{sD},{sQ},{sm})"
			results["SARIMA"] = ForecastResult(_sname, f, compute_metrics(test["price"], f), info, residuals=res)

	# ── Auto ARIMA ───────────────────────────────────────────────────────────
	if params["use_auto_arima"] and "Auto ARIMA" in selected:
		f, info, res = forecast_auto_arima_prices(
			train["price"], test["price"],
			seasonal=bool(params["auto_arima_seasonal"]), m=int(params["seasonal_period"]),
		)
		# info is already "Auto ARIMA(p, d, q)" — use it as the display name too
		results["Auto ARIMA"] = ForecastResult(info, f, compute_metrics(test["price"], f), info, residuals=res)

	# ── GARCH ────────────────────────────────────────────────────────────────
	if params["use_garch"] and "GARCH Volatility" in selected:
		f, info, res, vol = forecast_garch_volatility(
			train["price"], test.index, int(params["garch_p"]), int(params["garch_q"])
		)
		results["GARCH Volatility"] = ForecastResult(
			"GARCH Volatility", f, compute_metrics(test["price"], f), info, residuals=res, aux_series=vol
		)

	return results


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def render_results_table(results: Dict[str, ForecastResult]) -> pd.DataFrame:
	rows = [{"Model": r.name, **r.metrics, "Info": r.model_info} for r in results.values()]
	if not rows:
		return pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE", "R²", "Info"])
	df = pd.DataFrame(rows).sort_values(by=["MAPE", "RMSE", "MAE"], ascending=True).reset_index(drop=True)
	df.index += 1  # 1-based rank
	return df


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_selected_forecasts(
	train: pd.DataFrame, test: pd.DataFrame, results: Dict[str, ForecastResult],
	train_tail: int = 100,
) -> plt.Figure:
	# Show only the last `train_tail` rows of training data so the test window is magnified
	train_vis = train.tail(train_tail)

	fig, ax = plt.subplots(figsize=(15, 5.5))
	ax.fill_between(train_vis.index, train_vis["price"], alpha=0.08, color="#4f8ef7")
	ax.plot(train_vis.index, train_vis["price"], color="#4f8ef7", linewidth=1.2,
			label=f"Train (last {train_tail} days)", alpha=0.7)
	ax.plot(test.index, test["price"], color="#e8eaf0", linewidth=2, label="Actual (Test)")
	for i, result in enumerate(results.values()):
		ax.plot(
			result.forecast.index, result.forecast.values,
			linestyle="--", linewidth=1.7,
			color=FORECAST_PALETTE[i % len(FORECAST_PALETTE)],
			label=result.name,
		)
	ax.axvline(test.index[0], color="#4f6ef7", linestyle=":", linewidth=1.3, alpha=0.7)
	# Place the label just above the x-axis minimum of the visible window
	ymin, ymax = ax.get_ylim()
	ax.text(
		test.index[0], ymin + (ymax - ymin) * 0.01,
		"  Train / Test →", color="#4f6ef7",
		fontsize=7.5, va="bottom", ha="left", alpha=0.8,
	)
	ax.set_title("Forecast Comparison  (training window: last 100 days shown)", fontsize=12,
				 fontweight="bold", color="#d5d9eb")
	ax.set_xlabel("Date", labelpad=8)
	ax.set_ylabel("Price", labelpad=8)
	ax.legend(loc="upper left", ncols=3, fontsize=8)
	fig.tight_layout(pad=1.5)
	return fig


def plot_metric_bars(df: pd.DataFrame) -> plt.Figure:
	if df.empty:
		return plt.figure()
	n = len(df)
	colors = FORECAST_PALETTE[:n] if n <= len(FORECAST_PALETTE) else FORECAST_PALETTE * (n // len(FORECAST_PALETTE) + 1)

	fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
	for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
		bars = ax.bar(df["Model"], df[metric], color=colors[:n], edgecolor="#0d0f14", linewidth=0.6, zorder=3)
		# value labels on top of bars
		for bar in bars:
			h = bar.get_height()
			if not np.isnan(h):
				ax.text(bar.get_x() + bar.get_width() / 2, h * 1.015,
						f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color="#9da3bc")
		ax.set_title(metric, fontweight="bold", fontsize=10)
		ax.tick_params(axis="x", rotation=25)
		ax.set_xlabel("")
	fig.suptitle("Error Metrics by Model", fontweight="bold", fontsize=11, color="#d5d9eb", y=1.01)
	fig.tight_layout(pad=1.5)
	return fig


def plot_residuals(result: ForecastResult) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(14, 4))
	res = result.residuals.dropna() if result.residuals is not None else pd.Series(dtype=float)
	if res.empty:
		ax.text(0.5, 0.5, "No residuals available", ha="center", va="center", transform=ax.transAxes,
				color="#6e7490", fontsize=11)
		ax.axis("off")
		return fig
	ax.fill_between(res.index, res.values, alpha=0.15, color="#f7634f")
	ax.plot(res.index, res.values, color="#f7634f", linewidth=1, label="Residuals")
	ax.axhline(0, color="#e8eaf0", linestyle="--", linewidth=0.9, alpha=0.5)
	if len(res) >= 20:
		ax.plot(res.index, res.rolling(20).mean(), color="#4f8ef7", linewidth=2, label="20-day rolling mean")
	ax.set_title(f"Residuals — {result.name}", fontweight="bold")
	ax.set_xlabel("Date")
	ax.set_ylabel("Residual")
	ax.legend()
	fig.tight_layout(pad=1.5)
	return fig


def plot_diagnostics(result: ForecastResult) -> plt.Figure:
	fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
	res = result.residuals.dropna() if result.residuals is not None else pd.Series(dtype=float)
	if len(res) < 10:
		for ax in axes:
			ax.text(0.5, 0.5, "Not enough residuals for diagnostics",
					ha="center", va="center", transform=ax.transAxes, color="#6e7490")
			ax.axis("off")
		return fig
	lags = min(40, max(1, len(res) // 4))
	plot_acf(res,  zero=False, lags=lags, ax=axes[0])
	axes[0].set_title(f"ACF — {result.name}", fontweight="bold")
	plot_pacf(res, zero=False, lags=lags, ax=axes[1], method="ols")
	axes[1].set_title(f"PACF — {result.name}", fontweight="bold")
	fig.tight_layout(pad=1.5)
	return fig


def plot_garch_volatility(result: ForecastResult) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(14, 4))
	vol = result.aux_series.dropna() if result.aux_series is not None else pd.Series(dtype=float)
	if vol.empty:
		ax.text(0.5, 0.5, "No volatility series available",
				ha="center", va="center", transform=ax.transAxes, color="#6e7490")
		ax.axis("off")
		return fig
	ax.fill_between(vol.index, vol.values, alpha=0.18, color="#f7c34f")
	ax.plot(vol.index, vol.values, color="#f7c34f", linewidth=1.4, label="Conditional volatility")
	ax.set_title(f"GARCH Volatility — {result.name}", fontweight="bold")
	ax.set_xlabel("Date")
	ax.set_ylabel("Volatility")
	ax.legend()
	fig.tight_layout(pad=1.5)
	return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
	# ── Hero header ──────────────────────────────────────────────────────────
	st.markdown(
		"<div class='hero-title'>📈 Ticker Forecast Lab</div>"
		"<div class='hero-sub'>Time-series forecasting for equity indices & stocks — "
		"select a ticker, choose your models, tune parameters, and run.</div>",
		unsafe_allow_html=True,
	)

	if "forecast_state" not in st.session_state:
		st.session_state.forecast_state = None

	params = get_sidebar_params()

	# ── Run forecast ──────────────────────────────────────────────────────────
	if params["run"]:
		ticker_sym = params["ticker"]
		if not ticker_sym:
			st.error("Please enter or select a ticker before running.")
			return

		with st.spinner(f"Fetching data for **{ticker_sym}**…"):
			try:
				ticker_details = fetch_ticker_details(ticker_sym)
				cache = get_ticker_data_checkpoint_cache()
				ck = build_ticker_data_cache_key(ticker_sym, params["start_date"], params["end_date"])
				if ck in cache:
					raw = cache[ck]
				else:
					raw = download_ticker_data(ticker_sym, params["start_date"], params["end_date"])
					cache[ck] = raw
			except Exception as exc:
				st.error(f"❌ Failed to load **{params['ticker_label']}**: {exc}")
				return

		if len(raw) < 30:
			st.error("Not enough data to fit models. Try a longer date range.")
			return

		if len(raw) > int(params["recent_rows"]):
			raw = raw.tail(int(params["recent_rows"]))

		data  = add_log_returns(raw)
		train, test = train_test_split_frame(data, float(params["split_ratio"]))

		if test.empty:
			st.error("Train/test split produced an empty test set. Adjust the split ratio.")
			return

		n_models = len(params["selected_models"])
		with st.spinner(f"Fitting {n_models} model(s) — this may take a moment for SARIMA/GARCH…"):
			try:
				results = run_models(train, test, params)
			except Exception as exc:
				st.error(f"❌ Model fitting failed: {exc}")
				return

		if not results:
			st.warning("No models were executed. Select at least one model and press Run Forecast.")
			return

		st.session_state.forecast_state = {
			"params": params,
			"ticker_details": ticker_details,
			"raw": raw, "data": data, "train": train, "test": test,
			"results": results,
			"results_table": render_results_table(results),
		}

	# ── Empty state ───────────────────────────────────────────────────────────
	state = st.session_state.forecast_state
	if state is None:
		st.markdown(
			"<div class='empty-state'>"
			"<span class='icon'>🔬</span>"
			"<h2>No forecast yet</h2>"
			"<p>Pick a ticker, choose your models, tune parameters in the sidebar, "
			"then hit <strong>▶ Run Forecast</strong> to get started.</p>"
			"</div>",
			unsafe_allow_html=True,
		)
		return

	# ── Unpack state ──────────────────────────────────────────────────────────
	params        = state["params"]
	ticker_det    = state.get("ticker_details", {})
	data          = state["data"]
	train         = state["train"]
	test          = state["test"]
	results       = state["results"]
	results_table = state["results_table"]

	# ── Overview row ──────────────────────────────────────────────────────────
	ticker_name = ticker_det.get("name", params["ticker_label"])
	ticker_sym  = ticker_det.get("symbol", params["ticker"])
	st.markdown(
		f"<p class='section-header'>"
		f"{ticker_name}"
		f"<span class='ticker-badge'>{ticker_sym}</span>"
		f"<span class='model-count-badge'>{len(results)} model{'s' if len(results)!=1 else ''}</span>"
		f"</p>",
		unsafe_allow_html=True,
	)

	col_l, col_r = st.columns([1.1, 0.9])
	with col_l:
		st.markdown("<div class='callout-card'>", unsafe_allow_html=True)
		if ticker_det:
			st.dataframe(format_ticker_details(ticker_det), use_container_width=True, hide_index=True)
		st.write({
			"Total rows": len(data), "Train": len(train), "Test": len(test),
			"From": str(data.index.min().date()), "To": str(data.index.max().date()),
		})
		st.markdown("</div>", unsafe_allow_html=True)
	with col_r:
		st.markdown("<p style='font-family:DM Serif Display,serif;color:#c9ccda;font-size:1rem;margin-bottom:0.5rem'>Stationarity (ADF Test)</p>", unsafe_allow_html=True)
		st.dataframe(render_adf_summary(train["price"], train["returns"].dropna()),
					 use_container_width=True, hide_index=True)
		st.caption("ADF null hypothesis: the series has a unit root (non-stationary). "
				   "Reject H₀ if p < 0.05 to declare stationarity.")
		st.dataframe(data.tail(8), use_container_width=True)

	st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

	# ── Metric cards ──────────────────────────────────────────────────────────
	best = results_table.iloc[0] if not results_table.empty else None
	m1, m2, m3, m4 = st.columns(4)
	m1.metric("🏆 Best Model",  best["Model"] if best is not None else "—")
	m2.metric("📉 Best MAPE",   f"{best['MAPE']:.2f}%" if best is not None else "—")
	m3.metric("📏 Best RMSE",   f"{best['RMSE']:.2f}"  if best is not None else "—")
	m4.metric("🤖 Models Run",  len(results))

	st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

	# ── Tabs ──────────────────────────────────────────────────────────────────
	tab_forecasts, tab_metrics, tab_residuals, tab_diagnostics = st.tabs(
		["📈 Forecasts", "📊 Metrics", "🔴 Residuals", "🔬 Diagnostics"]
	)

	with tab_forecasts:
		st.pyplot(plot_selected_forecasts(train, test, results), clear_figure=True)
		if "GARCH Volatility" in results:
			st.pyplot(plot_garch_volatility(results["GARCH Volatility"]), clear_figure=True)
			st.caption("GARCH is a volatility model — the conditional volatility path is shown above. The price proxy is the last observed training price repeated flat.")
		combined = test[["price"]].copy()
		for r in results.values():
			combined[r.name] = r.forecast
		st.dataframe(combined.tail(30), use_container_width=True)
		st.download_button(
			"⬇ Download forecast CSV",
			data=combined.reset_index().to_csv(index=False).encode("utf-8"),
			file_name=f"{params['ticker']}_forecasts.csv",
			mime="text/csv",
		)

	with tab_metrics:
		st.pyplot(plot_metric_bars(results_table), clear_figure=True)
		st.caption("Models ranked by MAPE (ascending). Lower is better for MAE / RMSE / MAPE; higher is better for R².")
		st.dataframe(results_table, use_container_width=True)
		st.download_button(
			"⬇ Download metrics CSV",
			data=results_table.to_csv(index=False).encode("utf-8"),
			file_name=f"{params['ticker']}_metrics.csv",
			mime="text/csv",
		)

	with tab_residuals:
		st.caption(
			"Residuals = Actual − Forecast. Ideal residuals are centred around zero with no pattern over time. "
			"A non-zero rolling mean or fan-out shape signals a systematic bias in the model."
		)
		residual_choice = st.selectbox("Model", list(results.keys()), key="res_select")
		st.pyplot(plot_residuals(results[residual_choice]), clear_figure=True)
		if residual_choice == "GARCH Volatility":
			st.pyplot(plot_garch_volatility(results[residual_choice]), clear_figure=True)
			st.caption("GARCH is a volatility model — the conditional volatility path is more informative than the flat price proxy.")

	with tab_diagnostics:
		st.caption(
			"ACF measures correlation between the residuals at different lags. "
			"PACF isolates direct lag relationships. Spikes outside the confidence bands suggest "
			"unexploited autocorrelation structure — consider increasing p or q."
		)
		diag_choice = st.selectbox("Model", list(results.keys()), key="diag_select")
		st.pyplot(plot_diagnostics(results[diag_choice]), clear_figure=True)
		st.markdown(
			f"<div class='callout-card' style='margin-top:0.8rem'>"
			f"<span style='font-size:0.8rem;color:#6e7490;text-transform:uppercase;letter-spacing:0.6px;'>Model spec</span><br>"
			f"<span style='font-family:JetBrains Mono,monospace;font-size:0.88rem;color:#c6cbe8'>{results[diag_choice].model_info}</span>"
			f"</div>",
			unsafe_allow_html=True,
		)
		if diag_choice == "GARCH Volatility":
			st.pyplot(plot_garch_volatility(results[diag_choice]), clear_figure=True)

		st.markdown("<br>", unsafe_allow_html=True)
		st.markdown("<p style='font-family:DM Serif Display,serif;color:#c9ccda;font-size:1rem;'>Ljung-Box Test</p>", unsafe_allow_html=True)
		lj_res = results[diag_choice].residuals if results[diag_choice].residuals is not None else pd.Series(dtype=float)
		st.dataframe(render_ljung_box_summary(lj_res), use_container_width=True, hide_index=True)
		st.caption(
			"Ljung-Box null hypothesis: residuals have no autocorrelation (white noise). "
			"p < 0.05 → reject H₀ → autocorrelation remains → model may need higher orders."
		)

	# ── Summary table ─────────────────────────────────────────────────────────
	st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
	st.markdown("<p class='section-header'>Full Comparison Table</p>", unsafe_allow_html=True)
	st.dataframe(results_table, use_container_width=True)


if __name__ == "__main__":
	main()