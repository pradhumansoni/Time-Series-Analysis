from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Iterable, List, Optional, Tuple

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
except Exception:  # pragma: no cover - optional dependency
	auto_arima = None

try:
	from arch import arch_model
except Exception:  # pragma: no cover - optional dependency
	arch_model = None


st.set_page_config(page_title="Ticker Forecast Lab", layout="wide")


TICKER_OPTIONS: Dict[str, str] = {
	"NIFTY 50 Index (^NSEI)": "^NSEI",
	"NIFTY Bank (^NSEBANK)": "^NSEBANK",
	"BSE Sensex (^BSESN)": "^BSESN",
	"Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
	"TCS (TCS.NS)": "TCS.NS",
	"Infosys (INFY.NS)": "INFY.NS",
	"HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
	"ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
	"State Bank of India (SBIN.NS)": "SBIN.NS",
	"ITC (ITC.NS)": "ITC.NS",
}


def format_ticker_details(details: Dict[str, object]) -> pd.DataFrame:
	rows = []
	for label, key in [
		("Symbol", "symbol"),
		("Name", "name"),
		("Exchange", "exchange"),
		("Quote Type", "quoteType"),
		("Currency", "currency"),
		("Sector", "sector"),
		("Industry", "industry"),
		("Market Cap", "marketCap"),
		("Regular Market Price", "regularMarketPrice"),
		("Previous Close", "previousClose"),
		("Day Open", "open"),
		("Day High", "dayHigh"),
		("Day Low", "dayLow"),
		("52 Week Low", "fiftyTwoWeekLow"),
		("52 Week High", "fiftyTwoWeekHigh"),
	]:
		value = details.get(key)
		if value is None or value == "":
			continue
		if isinstance(value, (int, float, np.integer, np.floating)) and key in {"marketCap", "regularMarketPrice", "previousClose", "open", "dayHigh", "dayLow", "fiftyTwoWeekLow", "fiftyTwoWeekHigh"}:
			if key == "marketCap":
				value = f"{value:,.0f}"
			else:
				value = f"{float(value):,.2f}"
		rows.append({"Field": label, "Value": value})
	return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def fetch_ticker_details(ticker: str) -> Dict[str, object]:
	ticker = ticker.strip()
	if not ticker:
		raise ValueError("Please enter a ticker symbol.")

	ticker_obj = yf.Ticker(ticker)
	history = ticker_obj.history(period="5d", auto_adjust=True)
	if history is None or history.empty:
		raise ValueError(f"{ticker} does not appear to be a valid yfinance ticker.")

	info: Dict[str, object] = {}
	try:
		raw_info = ticker_obj.info or {}
		if isinstance(raw_info, dict):
			info = raw_info
	except Exception:
		info = {}

	latest = history.iloc[-1]
	return {
		"symbol": info.get("symbol", ticker),
		"name": info.get("longName") or info.get("shortName") or info.get("displayName") or ticker,
		"exchange": info.get("exchange") or info.get("fullExchangeName"),
		"quoteType": info.get("quoteType"),
		"currency": info.get("currency"),
		"sector": info.get("sector"),
		"industry": info.get("industry"),
		"marketCap": info.get("marketCap"),
		"regularMarketPrice": info.get("regularMarketPrice", float(latest.get("Close", np.nan))),
		"previousClose": info.get("previousClose", float(latest.get("Close", np.nan))),
		"open": info.get("open", float(latest.get("Open", latest.get("Close", np.nan)))),
		"dayHigh": info.get("dayHigh", float(latest.get("High", latest.get("Close", np.nan)))),
		"dayLow": info.get("dayLow", float(latest.get("Low", latest.get("Close", np.nan)))),
		"fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
		"fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
	}


def available_models() -> List[str]:
	models = ["Mean", "Naive", "Moving Average", "AR", "MA", "ARMA", "ARIMA", "SARIMA"]
	if auto_arima is not None:
		models.append("Auto ARIMA")
	if arch_model is not None:
		models.append("GARCH Volatility")
	return models


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
	return float(np.mean(np.abs((actual_aligned[non_zero] - predicted_aligned[non_zero]) / actual_aligned[non_zero])) * 100)


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
	actual_aligned, predicted_aligned = actual.align(predicted, join="inner")
	return {
		"MAE": float(mean_absolute_error(actual_aligned, predicted_aligned)),
		"RMSE": float(np.sqrt(mean_squared_error(actual_aligned, predicted_aligned))),
		"MAPE": safe_mape(actual_aligned, predicted_aligned),
		"R2": float(r2_score(actual_aligned, predicted_aligned)),
	}


def interpret_p_value(p_value: float, alpha: float, null_hypothesis: str, reject_message: str, fail_message: str) -> str:
	if p_value < alpha:
		return f"p-value < {alpha:.2f}, so reject H0: {null_hypothesis}. {reject_message}"
	return f"p-value >= {alpha:.2f}, so fail to reject H0: {null_hypothesis}. {fail_message}"


def get_rolling_checkpoint_cache() -> Dict[str, "ForecastResult"]:
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
	hashed_values = pd.util.hash_pandas_object(series, index=True).to_numpy(dtype=np.uint64).tobytes()
	return hashlib.sha256(hashed_values).hexdigest()


def build_rolling_cache_key(model_name: str, train_prices: pd.Series, test_prices: pd.Series, params: Dict[str, object]) -> str:
	cache_payload = {
		"model": model_name,
		"train": series_fingerprint(train_prices),
		"test": series_fingerprint(test_prices),
		"params": {key: value for key, value in params.items() if key != "run"},
	}
	return hashlib.sha256(json.dumps(cache_payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def download_ticker_data(ticker: str, start: str, end: str) -> pd.DataFrame:
	data = yf.download(
		tickers=ticker,
		start=start,
		end=end,
		interval="1d",
		auto_adjust=True,
		progress=False,
		threads=True,
	)

	if data is None or data.empty:
		raise ValueError(f"No data returned for {ticker}")

	frame = data.copy()
	if isinstance(frame.columns, pd.MultiIndex):
		if "Close" in frame.columns.get_level_values(0):
			close_frame = frame.xs("Close", axis=1, level=0)
		else:
			raise ValueError("Downloaded data does not contain a Close column.")
		if isinstance(close_frame, pd.DataFrame):
			if ticker in close_frame.columns:
				close_series = close_frame[ticker]
			else:
				close_series = close_frame.iloc[:, 0]
		else:
			close_series = close_frame
	else:
		if "Close" not in frame.columns:
			close_candidates = [column for column in frame.columns if str(column).lower() == "close"]
			if not close_candidates:
				raise ValueError("Downloaded data does not contain a Close column.")
			frame = frame.rename(columns={close_candidates[0]: "Close"})
		close_series = frame["Close"]

	frame = pd.DataFrame({"price": close_series})
	frame = frame.sort_index().asfreq("B").ffill().dropna()
	return frame


def add_log_returns(frame: pd.DataFrame) -> pd.DataFrame:
	result = frame.copy()
	result["returns"] = np.log(result["price"] / result["price"].shift(1))
	return result


def train_test_split_frame(data: pd.DataFrame, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
	split_idx = max(1, min(len(data) - 1, int(len(data) * split_ratio)))
	return data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()


def series_with_constant_value(value: float, index: pd.Index, name: str) -> pd.Series:
	return pd.Series(np.repeat(value, len(index)), index=index, name=name)


def mean_forecast(train_prices: pd.Series, test_index: pd.Index) -> pd.Series:
	return series_with_constant_value(train_prices.mean(), test_index, "mean_forecast")


def naive_forecast(train_prices: pd.Series, test_index: pd.Index) -> pd.Series:
	return series_with_constant_value(float(train_prices.iloc[-1]), test_index, "naive_forecast")


def moving_average_forecast(train_prices: pd.Series, test_index: pd.Index, window: int) -> pd.Series:
	window = max(1, min(window, len(train_prices)))
	forecast_value = float(train_prices.tail(window).mean())
	return series_with_constant_value(forecast_value, test_index, f"ma_{window}")


def return_order_to_string(order: Tuple[int, int, int]) -> str:
	return f"({order[0]}, {order[1]}, {order[2]})"


def price_path_from_returns(train_prices: pd.Series, forecast_returns: Iterable[float], index: pd.Index) -> pd.Series:
	current_price = float(train_prices.iloc[-1])
	forecast_values: List[float] = []
	for value in forecast_returns:
		current_price *= float(np.exp(value))
		forecast_values.append(current_price)
	return pd.Series(forecast_values, index=index)


def rolling_price_forecast(train_prices: pd.Series, test_prices: pd.Series, fit_callable, forecast_label: str, block_size: int) -> Tuple[pd.Series, pd.Series, str]:
	history = list(train_prices.dropna().tolist())
	forecast_values: List[float] = []
	forecast_index: List[pd.Timestamp] = []
	block_size = max(1, min(int(block_size), len(test_prices)))

	for start in range(0, len(test_prices), block_size):
		block_actual = test_prices.iloc[start : start + block_size]
		fitted = fit_callable(pd.Series(history, index=pd.RangeIndex(len(history))))
		block_forecast = fitted.forecast(steps=len(block_actual))
		forecast_values.extend(pd.Series(block_forecast).tolist())
		forecast_index.extend(block_actual.index.tolist())
		history.extend(block_actual.tolist())

	forecast_series = pd.Series(forecast_values, index=pd.Index(forecast_index), name=f"{forecast_label}_forecast")
	residuals = (test_prices - forecast_series).rename(f"{forecast_label}_forecast_residuals")
	return forecast_series, residuals, f"Rolling forecast ({block_size}-day blocks)"


def rolling_return_forecast(train_prices: pd.Series, test_prices: pd.Series, fit_callable, forecast_label: str, block_size: int) -> Tuple[pd.Series, pd.Series, str]:
	history = train_prices.dropna().tolist()
	forecast_values: List[float] = []
	forecast_index: List[pd.Timestamp] = []
	block_size = max(1, min(int(block_size), len(test_prices)))

	for start in range(0, len(test_prices), block_size):
		block_actual = test_prices.iloc[start : start + block_size]
		history_series = pd.Series(history)
		fitted = fit_callable(history_series)
		forecast_returns = fitted.forecast(steps=len(block_actual))
		block_forecast = price_path_from_returns(history_series, forecast_returns, block_actual.index)
		forecast_values.extend(block_forecast.tolist())
		forecast_index.extend(block_actual.index.tolist())
		history.extend(block_actual.tolist())

	forecast_series = pd.Series(forecast_values, index=pd.Index(forecast_index), name=f"{forecast_label}_forecast")
	residuals = (test_prices - forecast_series).rename(f"{forecast_label}_forecast_residuals")
	return forecast_series, residuals, f"Rolling forecast ({block_size}-day blocks)"


def forecast_return_based_model(
	train_prices: pd.Series,
	test_index: pd.Index,
	order: Tuple[int, int, int],
	model_name: str,
) -> Tuple[pd.Series, str, pd.Series]:
	return forecast_return_based_model_one_shot(train_prices, pd.Series(index=test_index, dtype=float), order, model_name)


def forecast_return_based_model_one_shot(
	train_prices: pd.Series,
	test_prices: pd.Series,
	order: Tuple[int, int, int],
	model_name: str,
) -> Tuple[pd.Series, str, pd.Series]:
	returns = np.log(train_prices).diff().dropna()
	fitted = ARIMA(
		returns,
		order=order,
		enforce_stationarity=False,
		enforce_invertibility=False,
	).fit()
	forecast_returns = fitted.forecast(steps=len(test_prices))
	forecast_prices = price_path_from_returns(train_prices, forecast_returns, test_prices.index)
	residuals = (test_prices - forecast_prices).rename(f"{model_name}_forecast_residuals")
	return forecast_prices, f"ARIMA{return_order_to_string(order)} on log returns", residuals


def forecast_arima_prices(
	train_prices: pd.Series,
	test_prices: pd.Series,
	order: Tuple[int, int, int],
) -> Tuple[pd.Series, str, pd.Series]:
	fitted = ARIMA(
		train_prices,
		order=order,
		enforce_stationarity=False,
		enforce_invertibility=False,
	).fit()
	forecast_prices = pd.Series(fitted.forecast(steps=len(test_prices)), index=test_prices.index)
	residuals = (test_prices - forecast_prices).rename(f"ARIMA{return_order_to_string(order)}_forecast_residuals")
	return forecast_prices, f"ARIMA{return_order_to_string(order)} on prices", residuals


def forecast_sarima_prices(
	train_prices: pd.Series,
	test_prices: pd.Series,
	order: Tuple[int, int, int],
	seasonal_order: Tuple[int, int, int, int],
) -> Tuple[pd.Series, str, pd.Series]:
	fitted = SARIMAX(
		train_prices,
		order=order,
		seasonal_order=seasonal_order,
		enforce_stationarity=False,
		enforce_invertibility=False,
	).fit(disp=False)
	forecast_prices = pd.Series(fitted.forecast(steps=len(test_prices)), index=test_prices.index)
	residuals = (test_prices - forecast_prices).rename("SARIMA_forecast_residuals")
	order_str = return_order_to_string(order)
	seasonal_str = f"({seasonal_order[0]}, {seasonal_order[1]}, {seasonal_order[2]}, {seasonal_order[3]})"
	return forecast_prices, f"SARIMA{order_str} x {seasonal_str}", residuals


def forecast_auto_arima_prices(
	train_prices: pd.Series,
	test_prices: pd.Series,
	seasonal: bool,
	m: int,
) -> Tuple[pd.Series, str, pd.Series]:
	if auto_arima is None:
		raise RuntimeError("pmdarima is not installed in this environment.")

	fitted = auto_arima(
		train_prices,
		seasonal=seasonal,
		m=max(1, m),
		stepwise=True,
		suppress_warnings=True,
		error_action="ignore",
		information_criterion="aic",
		max_p=6,
		max_q=6,
		max_d=2,
	)
	forecast_prices = pd.Series(fitted.predict(n_periods=len(test_prices)), index=test_prices.index)
	residuals = (test_prices - forecast_prices).rename("Auto_ARIMA_forecast_residuals")
	return forecast_prices, f"Auto ARIMA{fitted.order}", residuals


def forecast_garch_volatility(
	train_prices: pd.Series,
	test_index: pd.Index,
	p: int,
	q: int,
) -> Tuple[pd.Series, str, pd.Series, pd.Series]:
	if arch_model is None:
		raise RuntimeError("arch is not installed in this environment.")

	returns = 100 * np.log(train_prices / train_prices.shift(1)).dropna()
	fitted = arch_model(returns, vol="GARCH", p=p, q=q, mean="Zero", dist="normal").fit(disp="off")
	forecast = fitted.forecast(horizon=len(test_index), reindex=False)
	forecast_prices = series_with_constant_value(float(train_prices.iloc[-1]), test_index, "garch_volatility_proxy")
	residuals = pd.Series(fitted.resid, index=returns.index).dropna()
	in_sample_volatility = pd.Series(fitted.conditional_volatility, index=returns.index).dropna()
	out_of_sample_volatility = pd.Series(np.sqrt(forecast.variance.values[-1]), index=test_index, name="forecast_volatility")
	volatility_path = pd.concat([in_sample_volatility.rename("in_sample_volatility"), out_of_sample_volatility])
	return forecast_prices, f"GARCH({p}, {q}) volatility proxy", residuals, volatility_path


def render_adf_summary(price_series: pd.Series, return_series: pd.Series) -> pd.DataFrame:
	price_stat, price_pvalue, *_ = adfuller(price_series.dropna())
	return_stat, return_pvalue, *_ = adfuller(return_series.dropna())
	return pd.DataFrame(
		[
			{
				"Series": "Price",
				"ADF Statistic": price_stat,
				"p-value": price_pvalue,
				"Interpretation": interpret_p_value(
					price_pvalue,
					0.05,
					"the series has a unit root and is non-stationary",
					"our price series is stationary",
					"our price series is not stationary",
				),
			},
			{
				"Series": "Log Returns",
				"ADF Statistic": return_stat,
				"p-value": return_pvalue,
				"Interpretation": interpret_p_value(
					return_pvalue,
					0.05,
					"the series has a unit root and is non-stationary",
					"our log returns are stationary",
					"our log returns are not stationary",
				),
			},
		]
	)


def render_ljung_box_summary(residuals: pd.Series) -> pd.DataFrame:
	clean_residuals = residuals.dropna()
	if len(clean_residuals) < 10:
		return pd.DataFrame(
			[
				{
					"Test": "Ljung-Box",
					"p-value": np.nan,
					"Interpretation": "Not enough residual observations to run the test reliably.",
				}
			]
		)

	result = acorr_ljungbox(clean_residuals, lags=[10], return_df=True)
	p_value = float(result["lb_pvalue"].iloc[-1])
	interpretation = interpret_p_value(
		p_value,
		0.05,
		"the residuals are independently distributed (no autocorrelation)",
		"the residuals still have autocorrelation, so the model has not fully captured the structure",
		"the residuals look like white noise with no significant autocorrelation",
	)
	return pd.DataFrame(
		[
			{"Test": "Ljung-Box", "Lag": 10, "p-value": p_value, "Interpretation": interpretation},
		]
	)


def get_sidebar_params() -> Dict[str, object]:
	st.sidebar.header("Controls")
	st.sidebar.subheader("Ticker selection")
	ticker_source = st.sidebar.radio("Ticker source", ["Preset ticker", "Custom ticker"], index=0)
	if ticker_source == "Preset ticker":
		ticker_label = st.sidebar.selectbox("Ticker", list(TICKER_OPTIONS.keys()), index=0)
		ticker = TICKER_OPTIONS[ticker_label]
	else:
		ticker = st.sidebar.text_input("Custom ticker", value="", placeholder="Example: AAPL, MSFT, RELIANCE.NS").strip()
		ticker_label = ticker or "Custom ticker"

	start_date = st.sidebar.date_input("Start date", value=pd.Timestamp("2015-01-01"))
	end_date = st.sidebar.date_input("End date", value=pd.Timestamp.today().normalize())
	split_ratio = st.sidebar.slider("Train split", min_value=0.50, max_value=0.95, value=0.80, step=0.01)
	horizon = st.sidebar.slider("Forecast horizon", min_value=5, max_value=252, value=30, step=1)
	recent_rows = st.sidebar.slider(
		"Use only the most recent rows",
		min_value=250,
		max_value=4000,
		value=2500,
		step=50,
		help="Useful when a long history makes the higher-order models too slow.",
	)
	forecast_mode = st.sidebar.radio(
		"Forecast mode",
		["Entire test at once", "Rolling forecast"],
		index=0,
		help="Entire test at once forecasts the full test period in one shot. Rolling forecast refits on blocks of actual observations.",
	)
	rolling_window = st.sidebar.select_slider(
		"Rolling window size",
		options=[5, 10, 20],
		value=10,
		disabled=forecast_mode != "Rolling forecast",
		help="Used only when Rolling forecast is selected.",
	)

	st.sidebar.subheader("Models")
	selected_models = st.sidebar.multiselect("Choose models", available_models(), default=["Mean", "Naive", "Moving Average", "ARIMA"])

	st.sidebar.subheader("Baseline settings")
	ma_window = st.sidebar.slider("Moving average window", min_value=3, max_value=252, value=30, step=1)

	st.sidebar.subheader("AR / MA / ARMA")
	ar_p = st.sidebar.slider("AR order p", min_value=1, max_value=12, value=1, step=1)
	ma_q = st.sidebar.slider("MA order q", min_value=1, max_value=12, value=1, step=1)
	arma_p = st.sidebar.slider("ARMA p", min_value=1, max_value=12, value=1, step=1)
	arma_q = st.sidebar.slider("ARMA q", min_value=1, max_value=12, value=1, step=1)

	st.sidebar.subheader("ARIMA")
	arima_p = st.sidebar.slider("ARIMA p", min_value=0, max_value=12, value=1, step=1)
	arima_d = st.sidebar.slider("ARIMA d", min_value=0, max_value=2, value=1, step=1)
	arima_q = st.sidebar.slider("ARIMA q", min_value=0, max_value=12, value=1, step=1)

	st.sidebar.subheader("SARIMA")
	sarima_p = st.sidebar.slider("SARIMA p", min_value=0, max_value=6, value=1, step=1)
	sarima_d = st.sidebar.slider("SARIMA d", min_value=0, max_value=2, value=1, step=1)
	sarima_q = st.sidebar.slider("SARIMA q", min_value=0, max_value=6, value=1, step=1)
	sarima_P = st.sidebar.slider("Seasonal P", min_value=0, max_value=4, value=1, step=1)
	sarima_D = st.sidebar.slider("Seasonal D", min_value=0, max_value=2, value=1, step=1)
	sarima_Q = st.sidebar.slider("Seasonal Q", min_value=0, max_value=4, value=1, step=1)
	seasonal_period = st.sidebar.slider("Season length m", min_value=2, max_value=31, value=5, step=1)

	st.sidebar.subheader("Optional models")
	use_auto_arima = st.sidebar.checkbox("Enable Auto ARIMA", value=False, disabled=auto_arima is None)
	auto_arima_seasonal = st.sidebar.checkbox("Auto ARIMA seasonal", value=False, disabled=auto_arima is None)
	garch_p = st.sidebar.slider("GARCH p", min_value=1, max_value=5, value=1, step=1)
	garch_q = st.sidebar.slider("GARCH q", min_value=1, max_value=5, value=1, step=1)
	use_garch = st.sidebar.checkbox("Enable GARCH volatility proxy", value=False, disabled=arch_model is None)

	run = st.sidebar.button("Run forecast")

	return {
		"ticker": ticker,
		"ticker_label": ticker_label,
		"ticker_source": ticker_source,
		"start_date": str(start_date),
		"end_date": str(end_date),
		"split_ratio": split_ratio,
		"horizon": horizon,
		"recent_rows": recent_rows,
		"forecast_mode": forecast_mode,
		"rolling_window": rolling_window,
		"selected_models": selected_models,
		"ma_window": ma_window,
		"ar_p": ar_p,
		"ma_q": ma_q,
		"arma_p": arma_p,
		"arma_q": arma_q,
		"arima_p": arima_p,
		"arima_d": arima_d,
		"arima_q": arima_q,
		"sarima_p": sarima_p,
		"sarima_d": sarima_d,
		"sarima_q": sarima_q,
		"sarima_P": sarima_P,
		"sarima_D": sarima_D,
		"sarima_Q": sarima_Q,
		"seasonal_period": seasonal_period,
		"use_auto_arima": use_auto_arima,
		"auto_arima_seasonal": auto_arima_seasonal,
		"garch_p": garch_p,
		"garch_q": garch_q,
		"use_garch": use_garch,
		"run": run,
	}


def run_models(train: pd.DataFrame, test: pd.DataFrame, params: Dict[str, object]) -> Dict[str, ForecastResult]:
	selected = set(params["selected_models"])
	forecast_mode = str(params["forecast_mode"])
	rolling_window = int(params["rolling_window"])
	results: Dict[str, ForecastResult] = {}
	use_rolling = forecast_mode == "Rolling forecast"
	rolling_cache = get_rolling_checkpoint_cache()

	def rolling_cache_key(model_name: str) -> str:
		return build_rolling_cache_key(model_name, train["price"], test["price"], params)

	def get_cached_rolling_result(model_name: str) -> Optional[ForecastResult]:
		return rolling_cache.get(rolling_cache_key(model_name))

	def store_rolling_result(model_name: str, result: ForecastResult) -> ForecastResult:
		rolling_cache[rolling_cache_key(model_name)] = result
		return result

	if "Mean" in selected:
		forecast = mean_forecast(train["price"], test.index)
		results["Mean"] = ForecastResult(
			name="Mean",
			forecast=forecast,
			metrics=compute_metrics(test["price"], forecast),
			model_info="Constant mean forecast on prices",
			residuals=test["price"] - forecast,
		)

	if "Naive" in selected:
		forecast = naive_forecast(train["price"], test.index)
		results["Naive"] = ForecastResult(
			name="Naive",
			forecast=forecast,
			metrics=compute_metrics(test["price"], forecast),
			model_info="Last observed price repeated forward",
			residuals=test["price"] - forecast,
		)

	if "Moving Average" in selected:
		window = int(params["ma_window"])
		forecast = moving_average_forecast(train["price"], test.index, window)
		results["Moving Average"] = ForecastResult(
			name="Moving Average",
			forecast=forecast,
			metrics=compute_metrics(test["price"], forecast),
			model_info=f"Last {window} price average repeated forward",
			residuals=test["price"] - forecast,
		)

	if "AR" in selected:
		if use_rolling:
			cached_result = get_cached_rolling_result("AR")
			if cached_result is None:
				forecast, residuals, mode_info = rolling_return_forecast(
					train["price"],
					test["price"],
					lambda history_series: ARIMA(pd.Series(np.log(history_series).diff().dropna()), order=(int(params["ar_p"]), 0, 0), enforce_stationarity=False, enforce_invertibility=False).fit(),
					"AR",
					rolling_window,
				)
				cached_result = store_rolling_result(
					"AR",
					ForecastResult(
						name="AR",
						forecast=forecast,
						metrics=compute_metrics(test["price"], forecast),
						model_info=f"ARIMA({int(params['ar_p'])}, 0, 0) on log returns - {mode_info}",
						residuals=residuals,
					),
				)
			results["AR"] = cached_result
		else:
			forecast, model_info, residuals = forecast_return_based_model_one_shot(train["price"], test["price"], (int(params["ar_p"]), 0, 0), "AR")
			results["AR"] = ForecastResult(
				name="AR",
				forecast=forecast,
				metrics=compute_metrics(test["price"], forecast),
				model_info=model_info,
				residuals=residuals,
			)

	if "MA" in selected:
		if use_rolling:
			cached_result = get_cached_rolling_result("MA")
			if cached_result is None:
				forecast, residuals, mode_info = rolling_return_forecast(
					train["price"],
					test["price"],
					lambda history_series: ARIMA(pd.Series(np.log(history_series).diff().dropna()), order=(0, 0, int(params["ma_q"])), enforce_stationarity=False, enforce_invertibility=False).fit(),
					"MA",
					rolling_window,
				)
				cached_result = store_rolling_result(
					"MA",
					ForecastResult(
						name="MA",
						forecast=forecast,
						metrics=compute_metrics(test["price"], forecast),
						model_info=f"ARIMA(0, 0, {int(params['ma_q'])}) on log returns - {mode_info}",
						residuals=residuals,
					),
				)
			results["MA"] = cached_result
		else:
			forecast, model_info, residuals = forecast_return_based_model_one_shot(train["price"], test["price"], (0, 0, int(params["ma_q"])), "MA")
			results["MA"] = ForecastResult(
				name="MA",
				forecast=forecast,
				metrics=compute_metrics(test["price"], forecast),
				model_info=model_info,
				residuals=residuals,
			)

	if "ARMA" in selected:
		if use_rolling:
			cached_result = get_cached_rolling_result("ARMA")
			if cached_result is None:
				forecast, residuals, mode_info = rolling_return_forecast(
					train["price"],
					test["price"],
					lambda history_series: ARIMA(pd.Series(np.log(history_series).diff().dropna()), order=(int(params["arma_p"]), 0, int(params["arma_q"])), enforce_stationarity=False, enforce_invertibility=False).fit(),
					"ARMA",
					rolling_window,
				)
				cached_result = store_rolling_result(
					"ARMA",
					ForecastResult(
						name="ARMA",
						forecast=forecast,
						metrics=compute_metrics(test["price"], forecast),
						model_info=f"ARIMA({int(params['arma_p'])}, 0, {int(params['arma_q'])}) on log returns - {mode_info}",
						residuals=residuals,
					),
				)
			results["ARMA"] = cached_result
		else:
			forecast, model_info, residuals = forecast_return_based_model_one_shot(
				train["price"],
				test["price"],
				(int(params["arma_p"]), 0, int(params["arma_q"])),
				"ARMA",
			)
			results["ARMA"] = ForecastResult(
				name="ARMA",
				forecast=forecast,
				metrics=compute_metrics(test["price"], forecast),
				model_info=model_info,
				residuals=residuals,
			)

	if "ARIMA" in selected:
		if use_rolling:
			cached_result = get_cached_rolling_result("ARIMA")
			if cached_result is None:
				forecast, residuals, mode_info = rolling_price_forecast(
					train["price"],
					test["price"],
					lambda history_series: ARIMA(history_series, order=(int(params["arima_p"]), int(params["arima_d"]), int(params["arima_q"])), enforce_stationarity=False, enforce_invertibility=False).fit(),
					"ARIMA",
					rolling_window,
				)
				cached_result = store_rolling_result(
					"ARIMA",
					ForecastResult(
						name="ARIMA",
						forecast=forecast,
						metrics=compute_metrics(test["price"], forecast),
						model_info=f"ARIMA({int(params['arima_p'])}, {int(params['arima_d'])}, {int(params['arima_q'])}) on prices - {mode_info}",
						residuals=residuals,
					),
				)
			results["ARIMA"] = cached_result
		else:
			forecast, model_info, residuals = forecast_arima_prices(train["price"], test["price"], (int(params["arima_p"]), int(params["arima_d"]), int(params["arima_q"])))
			results["ARIMA"] = ForecastResult(
				name="ARIMA",
				forecast=forecast,
				metrics=compute_metrics(test["price"], forecast),
				model_info=model_info,
				residuals=residuals,
			)

	if "SARIMA" in selected:
		if use_rolling:
			cached_result = get_cached_rolling_result("SARIMA")
			if cached_result is None:
				forecast, residuals, mode_info = rolling_price_forecast(
					train["price"],
					test["price"],
					lambda history_series: SARIMAX(
						history_series,
						order=(int(params["sarima_p"]), int(params["sarima_d"]), int(params["sarima_q"])),
						seasonal_order=(int(params["sarima_P"]), int(params["sarima_D"]), int(params["sarima_Q"]), int(params["seasonal_period"])),
						enforce_stationarity=False,
						enforce_invertibility=False,
					).fit(disp=False),
					"SARIMA",
					rolling_window,
				)
				cached_result = store_rolling_result(
					"SARIMA",
					ForecastResult(
						name="SARIMA",
						forecast=forecast,
						metrics=compute_metrics(test["price"], forecast),
						model_info=f"SARIMA({int(params['sarima_p'])}, {int(params['sarima_d'])}, {int(params['sarima_q'])}) x ({int(params['sarima_P'])}, {int(params['sarima_D'])}, {int(params['sarima_Q'])}, {int(params['seasonal_period'])}) - {mode_info}",
						residuals=residuals,
					),
				)
			results["SARIMA"] = cached_result
		else:
			forecast, model_info, residuals = forecast_sarima_prices(
				train["price"],
				test["price"],
				(int(params["sarima_p"]), int(params["sarima_d"]), int(params["sarima_q"])),
				(int(params["sarima_P"]), int(params["sarima_D"]), int(params["sarima_Q"]), int(params["seasonal_period"])),
			)
			results["SARIMA"] = ForecastResult(
				name="SARIMA",
				forecast=forecast,
				metrics=compute_metrics(test["price"], forecast),
				model_info=model_info,
				residuals=residuals,
			)

	if params["use_auto_arima"] and "Auto ARIMA" in selected:
		forecast, model_info, residuals = forecast_auto_arima_prices(
			train["price"],
			test["price"],
			seasonal=bool(params["auto_arima_seasonal"]),
			m=int(params["seasonal_period"]),
		)
		results["Auto ARIMA"] = ForecastResult(
			name="Auto ARIMA",
			forecast=forecast,
			metrics=compute_metrics(test["price"], forecast),
			model_info=model_info,
			residuals=residuals,
		)

	if params["use_garch"] and "GARCH Volatility" in selected:
		forecast, model_info, residuals, volatility_path = forecast_garch_volatility(
			train["price"],
			test.index,
			int(params["garch_p"]),
			int(params["garch_q"]),
		)
		results["GARCH Volatility"] = ForecastResult(
			name="GARCH Volatility",
			forecast=forecast,
			metrics=compute_metrics(test["price"], forecast),
			model_info=model_info,
			residuals=residuals,
			aux_series=volatility_path,
		)

	return results


def render_results_table(results: Dict[str, ForecastResult]) -> pd.DataFrame:
	rows = [{"Model": result.name, **result.metrics, "Info": result.model_info} for result in results.values()]
	if not rows:
		return pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE", "R2", "Info"])
	return pd.DataFrame(rows).sort_values(by=["MAPE", "RMSE", "MAE"], ascending=True).reset_index(drop=True)


def plot_selected_forecasts(train: pd.DataFrame, test: pd.DataFrame, results: Dict[str, ForecastResult]) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(15, 6))
	ax.plot(train.index, train["price"], color="steelblue", linewidth=1, label="Train")
	ax.plot(test.index, test["price"], color="black", linewidth=2, label="Actual")
	palette = ["#b80c09", "#6a4c93", "#f48c06", "#1d3557", "#2a9d8f", "#e76f51", "#7f5539", "#457b9d"]
	for idx, result in enumerate(results.values()):
		ax.plot(result.forecast.index, result.forecast.values, linestyle="--", linewidth=1.5, color=palette[idx % len(palette)], label=result.name)
	ax.axvline(test.index[0], color="gray", linestyle=":", linewidth=1)
	ax.set_title("Forecast comparison", fontweight="bold")
	ax.set_xlabel("Date")
	ax.set_ylabel("Price")
	ax.legend(loc="upper left", ncols=2)
	fig.tight_layout()
	return fig


def plot_metric_bars(results_table: pd.DataFrame) -> plt.Figure:
	fig, axes = plt.subplots(1, 3, figsize=(18, 5))
	if results_table.empty:
		return fig
	colors = plt.cm.Dark2(np.linspace(0, 1, len(results_table)))
	for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
		ax.bar(results_table["Model"], results_table[metric], color=colors, edgecolor="white")
		ax.set_title(metric, fontweight="bold")
		ax.tick_params(axis="x", rotation=30)
		ax.grid(axis="y", alpha=0.2)
	fig.suptitle("Error metric comparison", fontweight="bold")
	fig.tight_layout()
	return fig


def plot_residuals(result: ForecastResult) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(14, 5))
	residuals = result.residuals.dropna() if result.residuals is not None else pd.Series(dtype=float)
	if residuals.empty:
		ax.text(0.5, 0.5, "No residuals available", ha="center", va="center", transform=ax.transAxes)
		ax.axis("off")
		return fig
	ax.plot(residuals.index, residuals.values, color="red", label="Residuals")
	if len(residuals) >= 20:
		ax.plot(residuals.index, residuals.rolling(20).mean(), color="blue", linewidth=2, label="20-day rolling mean")
	ax.axhline(0, color="black", linestyle="--")
	ax.set_title(f"Residuals for {result.name}", fontweight="bold")
	ax.set_xlabel("Date")
	ax.set_ylabel("Residual")
	ax.legend()
	fig.tight_layout()
	return fig


def plot_diagnostics(result: ForecastResult) -> plt.Figure:
	fig, axes = plt.subplots(1, 2, figsize=(16, 5))
	residuals = result.residuals.dropna() if result.residuals is not None else pd.Series(dtype=float)
	if len(residuals) < 10:
		axes[0].text(0.5, 0.5, "Not enough residuals for diagnostics", ha="center", va="center", transform=axes[0].transAxes)
		axes[0].axis("off")
		axes[1].axis("off")
		return fig
	lags = min(40, max(1, len(residuals) // 4))
	plot_acf(residuals, zero=False, lags=lags, ax=axes[0])
	axes[0].set_title(f"ACF - {result.name}")
	plot_pacf(residuals, zero=False, lags=lags, ax=axes[1], method="ols")
	axes[1].set_title(f"PACF - {result.name}")
	fig.tight_layout()
	return fig


def plot_garch_volatility(result: ForecastResult) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(14, 5))
	volatility = result.aux_series.dropna() if result.aux_series is not None else pd.Series(dtype=float)
	if volatility.empty:
		ax.text(0.5, 0.5, "No volatility series available", ha="center", va="center", transform=ax.transAxes)
		ax.axis("off")
		return fig

	ax.plot(volatility.index, volatility.values, color="darkorange", linewidth=1.5, label="Conditional volatility")
	ax.set_title(f"GARCH volatility path - {result.name}", fontweight="bold")
	ax.set_xlabel("Date")
	ax.set_ylabel("Volatility")
	ax.legend()
	fig.tight_layout()
	return fig


def main() -> None:
	st.title("Ticker Forecast Lab")
	st.caption("Streamlit version of the forecasting notebook with ticker selection, model selection, and tunable statistical methods.")

	if "forecast_state" not in st.session_state:
		st.session_state.forecast_state = None

	params = get_sidebar_params()

	st.sidebar.markdown("---")
	st.sidebar.markdown("The selected models run only after you press Run forecast. The last forecast stays visible until you run a new one.")

	if params["run"]:
		try:
			ticker_details = fetch_ticker_details(params["ticker"])
			ticker_cache = get_ticker_data_checkpoint_cache()
			cache_key = build_ticker_data_cache_key(params["ticker"], params["start_date"], params["end_date"])
			if cache_key in ticker_cache:
				raw = ticker_cache[cache_key]
			else:
				raw = download_ticker_data(params["ticker"], params["start_date"], params["end_date"])
				ticker_cache[cache_key] = raw
		except Exception as exc:
			st.error(f"Failed to load {params['ticker_label']}: {exc}")
			return

		if len(raw) < 30:
			st.error("Not enough data to fit the selected models. Try a longer date range.")
			return

		if len(raw) > int(params["recent_rows"]):
			raw = raw.tail(int(params["recent_rows"]))

		data = add_log_returns(raw)
		train, test = train_test_split_frame(data, float(params["split_ratio"]))

		if test.empty:
			st.error("Train/test split produced an empty test set. Adjust the split ratio.")
			return

		try:
			results = run_models(train, test, params)
		except Exception as exc:
			st.error(f"Model fitting failed: {exc}")
			return

		if not results:
			st.warning("No models were executed. Make sure at least one selected model is enabled.")
			return

		st.session_state.forecast_state = {
			"params": params,
			"ticker_details": ticker_details,
			"raw": raw,
			"data": data,
			"train": train,
			"test": test,
			"results": results,
			"results_table": render_results_table(results),
		}

	state = st.session_state.forecast_state
	if state is None:
		st.info("Choose a ticker, select the models you want to compare, tune their parameters, and press Run forecast.")
		return

	params = state["params"]
	ticker_details = state.get("ticker_details", {})
	raw = state["raw"]
	data = state["data"]
	train = state["train"]
	test = state["test"]
	results = state["results"]
	results_table = state["results_table"]

	left, right = st.columns([1.1, 0.9])
	with left:
		st.subheader(f"{params['ticker_label']} overview")
		if ticker_details:
			st.dataframe(format_ticker_details(ticker_details), use_container_width=True, hide_index=True)
		st.write(
			{
				"Rows": len(data),
				"Train rows": len(train),
				"Test rows": len(test),
				"Start": str(data.index.min().date()),
				"End": str(data.index.max().date()),
			}
		)
		st.dataframe(data.tail(10), use_container_width=True)
	with right:
		st.subheader("Stationarity check")
		st.dataframe(render_adf_summary(train["price"], train["returns"].dropna()), use_container_width=True)
		st.caption("ADF null hypothesis: the series has a unit root. If p-value < 0.05, we reject H0 and treat the series as stationary.")
	best_row = results_table.iloc[0] if not results_table.empty else None

	summary_col1, summary_col2, summary_col3 = st.columns(3)
	summary_col1.metric("Best model by MAPE", best_row["Model"] if best_row is not None else "N/A")
	summary_col2.metric("Best MAPE", f"{best_row['MAPE']:.2f}%" if best_row is not None else "N/A")
	summary_col3.metric("Models run", len(results))

	combined = test[["price"]].copy()
	for result in results.values():
		combined[result.name] = result.forecast

	tabs = st.tabs(["Forecasts", "Metrics", "Residuals", "Diagnostics"])

	with tabs[0]:
		st.pyplot(plot_selected_forecasts(train, test, results), clear_figure=True)
		st.dataframe(combined, use_container_width=True)
		st.download_button(
			"Download forecast comparison CSV",
			data=combined.reset_index().to_csv(index=False).encode("utf-8"),
			file_name=f"{params['ticker']}_forecast_comparison.csv",
			mime="text/csv",
		)

	with tabs[1]:
		st.pyplot(plot_metric_bars(results_table), clear_figure=True)
		st.dataframe(results_table, use_container_width=True)
		st.download_button(
			"Download metrics CSV",
			data=results_table.to_csv(index=False).encode("utf-8"),
			file_name=f"{params['ticker']}_forecast_metrics.csv",
			mime="text/csv",
		)

	with tabs[2]:
		residual_choice = st.selectbox("Residual series", list(results.keys()))
		st.pyplot(plot_residuals(results[residual_choice]), clear_figure=True)
		if residual_choice == "GARCH Volatility":
			st.pyplot(plot_garch_volatility(results[residual_choice]), clear_figure=True)
			st.caption("GARCH is a volatility model, so the most useful visual is the conditional volatility path rather than a flat price forecast line.")

	with tabs[3]:
		diag_choice = st.selectbox("Model for ACF/PACF diagnostics", list(results.keys()), key="diag_choice")
		st.pyplot(plot_diagnostics(results[diag_choice]), clear_figure=True)
		st.write(results[diag_choice].model_info)
		if diag_choice == "GARCH Volatility":
			st.pyplot(plot_garch_volatility(results[diag_choice]), clear_figure=True)
		st.subheader("Residual autocorrelation test")
		st.dataframe(render_ljung_box_summary(results[diag_choice].residuals if results[diag_choice].residuals is not None else pd.Series(dtype=float)), use_container_width=True)
		st.caption("Ljung-Box null hypothesis: residuals have no autocorrelation. If p-value < 0.05, we reject H0 and say the residuals are still autocorrelated.")

	st.markdown("---")
	st.subheader("Comparison table")
	st.dataframe(results_table, use_container_width=True)


if __name__ == "__main__":
	main()
