#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tungsten APT Dashboard (Streamlit)

- Scrapes/updates APT 88.5% spot history from ScrapMonster (EXW China / FOB China / Warehouse Rotterdam)
- Builds a tungsten thematic equity basket from global listed names
- Aligns equity basket ONLY to dates where spot exists (spot dates define the index start)
- Shows daily/weekly/monthly changes for spot + basket
"""

import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.graph_objects as go


# =============================
# CONFIG
# =============================

DATA_DIR = "data"
PLOTS_DIR = "plots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

APT_URL = "https://www.scrapmonster.com/metal-prices/tungsten-apt-885-min-price/816"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_TIMEOUT = 25
RETRIES = 3
RETRY_SLEEP_SEC = 2

# ScrapMonster APT page typically shows these 3 series (in order)
SERIES_LABELS_IN_ORDER = ["EXW China", "FOB China", "Warehouse Rotterdam"]

TUNGSTEN_TICKERS_DEFAULT = [
    # Direct tungsten exposure
    "ALM",        # Almonty Industries
    "TUN.L",      # Tungsten West
    "TGN.AX",     # Tungsten Mining NL
    "6998.T",     # Nippon Tungsten Co., Ltd.

    # China/APAC exposure
    "600549.SS",  # Xiamen Tungsten
    "002378.SZ",  # Chongyi Zhangyuan Tungsten
    "03993.HK",   # CMOC Group
    "603993.SS",  # CMOC Group (Shanghai)
    "000657.SZ",  # China Tungsten & High-tech Materials

    # Materials / powders exposure
    "UMI.BR",     # Umicore
    "5711.T",     # Mitsubishi Materials
    "SAND.ST",    # Sandvik AB
    "5802.T",     # Sumitomo Electric Industries
]

TICKER_NAME_MAP = {
    "ALM": "Almonty Industries",
    "TUN.L": "Tungsten West",
    "TGN.AX": "Tungsten Mining NL",
    "6998.T": "Nippon Tungsten",
    "600549.SS": "Xiamen Tungsten",
    "002378.SZ": "Chongyi Zhangyuan Tungsten",
    "03993.HK": "CMOC Group (HK)",
    "603993.SS": "CMOC Group (SH)",
    "000657.SZ": "China Tungsten & HT Materials",
    "UMI.BR": "Umicore",
    "5711.T": "Mitsubishi Materials",
    "SAND.ST": "Sandvik AB",
    "5802.T": "Sumitomo Electric",
}

TICKER_ALIAS_MAP = {
    # Yahoo sometimes uses 4-digit HK tickers without leading zero
    "03993.HK": "3993.HK",
}

CHART_COLORS = {
    "spot": "#f59e0b",
    "basket": "#e5e7eb",
    "composite": "#22c55e",
    "grid": "#2b3442",
    "text": "#e5e7eb",
    "tick": "#cbd5f5",
    "spine": "#334155",
    "bg": "#0f172a",
    "panel": "#111827",
}


# =============================
# HELPERS
# =============================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = re.sub(r"[^0-9\.\-]", "", str(x))
    if s in ("", "-", ".", "-.", ".-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_usdcny_rate_yf(period: str = "5d") -> Optional[float]:
    """
    Returns USD/CNY (i.e., how many CNY per 1 USD) using Yahoo Finance.
    Tries a couple common FX tickers and returns the most recent close.
    """
    candidates = ["CNY=X", "USDCNY=X"]
    for t in candidates:
        try:
            df = yf.download(t, period=period, progress=False)
            if df is None or df.empty:
                continue
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            val = float(close.dropna().iloc[-1])
            if val > 3:
                return val
        except Exception:
            continue
    return None


def modeled_buy_price_series_from_china_benchmark(
    spot_series_usd_mt: pd.Series,
    china_cny_per_tonne: float,
    usdcny: float,
    incoterm_uplift_pct: float = 0.0,
    buyer_premium_pct: float = 0.0,
    blend_with_spot: float = 0.0,
) -> Tuple[pd.Series, float]:
    """
    Builds a proxy 'buy price' series in USD/MT anchored to a China benchmark.
    It level-shifts the ScrapMonster spot series to the China benchmark level.
    """
    spot = spot_series_usd_mt.dropna().copy()
    if spot.empty:
        return pd.Series(dtype="float64"), float("nan")

    base_usd = float(china_cny_per_tonne) / float(usdcny)
    adj_mult = (1.0 + incoterm_uplift_pct / 100.0) * (1.0 + buyer_premium_pct / 100.0)
    buy_level_today = base_usd * adj_mult

    last_spot = float(spot.iloc[-1])
    ratio = (buy_level_today / last_spot) if last_spot != 0 else 1.0
    shifted = (spot * ratio).rename("Buy_Price_USD_MT")

    if blend_with_spot and blend_with_spot > 0:
        a = float(blend_with_spot)
        shifted = (a * spot + (1.0 - a) * shifted).rename("Buy_Price_USD_MT")

    return shifted, buy_level_today




def fetch_html(url: str) -> str:
    last_err = None
    for i in range(RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP_SEC * (i + 1))
    raise RuntimeError(f"Failed to fetch {url}. Last error: {last_err}")


def to_pct_change_from_start(s: pd.Series) -> pd.Series:
    s = s.dropna()
    return (s / s.iloc[0] - 1.0) * 100.0


def build_composite_pct(spot: pd.Series, basket: pd.Series, w_spot: float = 0.5) -> pd.Series:
    w_basket = 1.0 - w_spot
    spot_pct = to_pct_change_from_start(spot)
    basket_pct = to_pct_change_from_start(basket)
    common = spot_pct.index.intersection(basket_pct.index)
    spot_pct = spot_pct.loc[common]
    basket_pct = basket_pct.loc[common]
    comp = w_spot * spot_pct + w_basket * basket_pct
    comp.name = "Composite_%_change"
    return comp


def parse_apt_page(html: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      latest_df: one row per series with latest price + date (best-effort)
      hist_df:   stacked historical rows with columns [series, date, price, unit, pulled_at_utc]
    """
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]

    price_idxs = [i for i, ln in enumerate(lines) if "$US/MT" in ln and to_float(ln) is not None]
    latest_rows = []
    for j, idx in enumerate(price_idxs[:3]):
        price = to_float(lines[idx])

        dt = None
        for k in range(idx + 1, min(idx + 8, len(lines))):
            try:
                dt_candidate = pd.to_datetime(lines[k], errors="raise")
                dt = dt_candidate.date().isoformat()
                break
            except Exception:
                continue

        label = SERIES_LABELS_IN_ORDER[j] if j < len(SERIES_LABELS_IN_ORDER) else f"Series_{j+1}"
        latest_rows.append({
            "series": label,
            "latest_price": price,
            "latest_unit": "USD/MT",
            "latest_date": dt,
            "pulled_at_utc": now_utc_iso()
        })
    latest_df = pd.DataFrame(latest_rows)

    tables = pd.read_html(html)
    hist_tables = []
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if ("date" in cols) and ("price" in cols) and ("unit" in cols):
            hist_tables.append(t.copy())

    out = []
    for i, t in enumerate(hist_tables[:3]):
        label = SERIES_LABELS_IN_ORDER[i] if i < len(SERIES_LABELS_IN_ORDER) else f"Series_{i+1}"
        t.columns = ["date", "price", "unit"]
        t["series"] = label
        t["price"] = t["price"].apply(to_float)
        t["date"] = pd.to_datetime(t["date"], errors="coerce").dt.date.astype(str)
        t["pulled_at_utc"] = now_utc_iso()
        out.append(t[["series", "date", "price", "unit", "pulled_at_utc"]])

    if out:
        hist_df = pd.concat(out, ignore_index=True).dropna(subset=["date", "price"])
    else:
        hist_df = pd.DataFrame(columns=["series", "date", "price", "unit", "pulled_at_utc"])

    return latest_df, hist_df


def update_apt_files(latest_df: pd.DataFrame, hist_df: pd.DataFrame) -> Tuple[str, str]:
    latest_path = os.path.join(DATA_DIR, "apt_latest.csv")
    hist_path = os.path.join(DATA_DIR, "apt_history.csv")

    if not latest_df.empty:
        latest_df.to_csv(latest_path, index=False)

    if os.path.exists(hist_path):
        old = pd.read_csv(hist_path)
        combined = pd.concat([old, hist_df], ignore_index=True)
    else:
        combined = hist_df.copy()

    combined = combined.drop_duplicates(subset=["series", "date", "price"], keep="first")
    combined.to_csv(hist_path, index=False)

    return latest_path, hist_path


def _download_one(ticker: str, start: Optional[str] = None, period: Optional[str] = None) -> pd.Series:
    data = yf.download(ticker, start=start, period=period, progress=False)
    if data is None or data.empty:
        return pd.Series(dtype="float64", name=ticker)
    if "Close" not in data.columns:
        return pd.Series(dtype="float64", name=ticker)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            close = close[ticker]
        elif close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            return pd.Series(dtype="float64", name=ticker)
    s = close.rename(ticker)
    return s


def download_equities(tickers: List[str], start_date: str) -> pd.DataFrame:
    frames = []
    for t in tickers:
        s = _download_one(t, start=start_date)
        if s.dropna().empty:
            alias = TICKER_ALIAS_MAP.get(t)
            if alias:
                s = _download_one(alias, start=start_date).rename(t)
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    px = pd.concat(frames, axis=1)
    px = px.dropna(how="all")
    px = px.dropna(axis=1, how="all")
    return px


@st.cache_data(show_spinner=False)
def download_equities_5y(tickers: List[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        # Use max history for stability, then slice last 5 years.
        s = _download_one(t, period="max")
        if not s.dropna().empty:
            end_date = s.index.max()
            start_date = end_date - pd.DateOffset(years=5)
            s = s.loc[s.index >= start_date]
        if s.dropna().empty:
            alias = TICKER_ALIAS_MAP.get(t)
            if alias:
                s = _download_one(alias, period="max").rename(t)
                if not s.dropna().empty:
                    end_date = s.index.max()
                    start_date = end_date - pd.DateOffset(years=5)
                    s = s.loc[s.index >= start_date]
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    px = pd.concat(frames, axis=1)
    px = px.dropna(how="all")
    px = px.dropna(axis=1, how="all")
    return px


def build_equal_weight_index_on_spot_dates(px_daily: pd.DataFrame, spot_dates: pd.DatetimeIndex) -> pd.Series:
    px_ffill = px_daily.sort_index().ffill()
    rets = px_ffill.pct_change(fill_method=None).fillna(0.0)

    w = pd.Series(1.0 / rets.shape[1], index=rets.columns)
    idx_daily = (1.0 + (rets * w).sum(axis=1)).cumprod()
    idx_daily.name = "Tungsten_APT_Thematic_Basket"

    idx_on_spot = idx_daily.reindex(spot_dates, method="ffill").dropna()
    idx_on_spot = 100.0 * (idx_on_spot / idx_on_spot.iloc[0])
    idx_on_spot.name = "Tungsten_APT_Thematic_Basket_100_start"
    return idx_on_spot


def load_spot_series(hist_path: str) -> pd.DataFrame:
    df = pd.read_csv(hist_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df.sort_values("date")

    wide = df.pivot_table(index="date", columns="series", values="price", aggfunc="mean").sort_index()

    for c in SERIES_LABELS_IN_ORDER:
        if c not in wide.columns:
            wide[c] = pd.NA

    wide["APT_AVG"] = wide[SERIES_LABELS_IN_ORDER].mean(axis=1, skipna=True)
    return wide


def normalize_prices_on_spot_dates(px_daily: pd.DataFrame, spot_dates: pd.DatetimeIndex) -> pd.DataFrame:
    px_ffill = px_daily.sort_index().ffill()
    px_on_spot = px_ffill.reindex(spot_dates, method="ffill").dropna(how="all")
    if px_on_spot.empty:
        return px_on_spot
    norm = 100.0 * (px_on_spot / px_on_spot.iloc[0])
    return norm


def normalize_prices(px_daily: pd.DataFrame) -> pd.DataFrame:
    px_ffill = px_daily.sort_index().ffill()
    if px_ffill.empty:
        return px_ffill
    norm = 100.0 * (px_ffill / px_ffill.iloc[0])
    return norm


def normalize_prices_window(px_window: pd.DataFrame) -> pd.DataFrame:
    if px_window.empty:
        return px_window
    px_sorted = px_window.sort_index()
    def _norm_col(s: pd.Series) -> pd.Series:
        s = s.dropna()
        if s.empty:
            return s
        base = s.iloc[0]
        return 100.0 * (s / base)
    norm = px_sorted.apply(_norm_col, axis=0)
    return norm


def build_equal_weight_index_daily(px_daily: pd.DataFrame) -> pd.Series:
    px_ffill = px_daily.sort_index().ffill()
    rets = px_ffill.pct_change(fill_method=None).fillna(0.0)
    w = pd.Series(1.0 / rets.shape[1], index=rets.columns)
    idx_daily = (1.0 + (rets * w).sum(axis=1)).cumprod()
    idx_daily = 100.0 * (idx_daily / idx_daily.iloc[0])
    idx_daily.name = "Tungsten_APT_Thematic_Basket_Daily_100_start"
    return idx_daily


def fetch_intraday_last_and_prev_close(tickers: List[str]) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (last_price, prev_close) for each ticker when available.
    Falls back to daily close if intraday is unavailable.
    """
    if not tickers:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")

    last_price = pd.Series(index=tickers, dtype="float64")
    prev_close = pd.Series(index=tickers, dtype="float64")

    try:
        daily = yf.download(tickers, period="5d", interval="1d", progress=False)["Close"]
        if isinstance(daily, pd.Series):
            daily = daily.to_frame()
    except Exception:
        daily = pd.DataFrame()

    try:
        intraday = yf.download(tickers, period="1d", interval="1m", progress=False)["Close"]
        if isinstance(intraday, pd.Series):
            intraday = intraday.to_frame()
    except Exception:
        intraday = pd.DataFrame()

    for t in tickers:
        last_val = np.nan
        prev_val = np.nan

        if not intraday.empty and t in intraday.columns:
            s = intraday[t].dropna()
            if not s.empty:
                last_val = float(s.iloc[-1])

        if not daily.empty and t in daily.columns:
            s = daily[t].dropna()
            if len(s) >= 2:
                prev_val = float(s.iloc[-2])
            elif len(s) == 1:
                prev_val = float(s.iloc[-1])

        if np.isnan(last_val) and not daily.empty and t in daily.columns:
            s = daily[t].dropna()
            if not s.empty:
                last_val = float(s.iloc[-1])

        last_price[t] = last_val
        prev_close[t] = prev_val

    return last_price, prev_close


def ticker_label(ticker: str) -> str:
    return TICKER_NAME_MAP.get(ticker, ticker)


def align_equities_to_spot(px_daily: pd.DataFrame, spot_index: pd.DatetimeIndex) -> pd.DataFrame:
    px_ffill = px_daily.sort_index().ffill()
    px_on_spot = px_ffill.reindex(spot_index, method="ffill")
    return px_on_spot


def normalize_from_start(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base = df.iloc[0]
    return 100.0 * (df / base)


def to_monthly_last(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    return s.resample("M").last()


def to_monthly_last_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.resample("M").last()


def fit_linear_model(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[np.ndarray, float, pd.Series, pd.Series, pd.Series]:
    """
    Returns (coef, intercept, y_hat, X_mean, X_std) using standardized X.
    """
    X = X.dropna()
    y = y.loc[X.index].dropna()
    X = X.loc[y.index]
    if X.empty or y.empty:
        return np.array([]), 0.0, pd.Series(dtype="float64"), pd.Series(dtype="float64"), pd.Series(dtype="float64")

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0).replace(0, np.nan)
    Xz = (X - X_mean) / X_std
    Xz = Xz.fillna(0.0)

    X_mat = np.column_stack([np.ones(len(Xz)), Xz.values])
    beta = np.linalg.lstsq(X_mat, y.values, rcond=None)[0]
    intercept = float(beta[0])
    coef = beta[1:]

    y_hat = pd.Series(X_mat @ beta, index=X.index, name="Modeled_Spot")
    return coef, intercept, y_hat, X_mean, X_std


def model_metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float, float]:
    y_true = y_true.dropna()
    y_pred = y_pred.reindex(y_true.index).dropna()
    y_true = y_true.loc[y_pred.index]
    if y_true.empty or y_pred.empty:
        return np.nan, np.nan, np.nan
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    rmse = float(np.sqrt((ss_res / len(y_true))))
    corr = float(y_true.corr(y_pred))
    return r2, rmse, corr


def last_value_on_or_before(s: pd.Series, target_date: pd.Timestamp) -> Optional[float]:
    s = s.dropna()
    if s.empty:
        return None
    s = s.loc[s.index <= target_date]
    if s.empty:
        return None
    return float(s.iloc[-1])


def compute_change(series: pd.Series, days_back: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (absolute_change, pct_change) using the last value
    and the last value on or before (last_date - days_back).
    """
    series = series.dropna()
    if series.empty:
        return None, None

    last_date = series.index[-1]
    last_value = float(series.iloc[-1])
    ref_date = last_date - timedelta(days=days_back)
    ref_value = last_value_on_or_before(series, ref_date)
    if ref_value is None or ref_value == 0:
        return None, None

    abs_change = last_value - ref_value
    pct_change = (last_value / ref_value - 1.0) * 100.0
    return abs_change, pct_change


def compute_months_change(series: pd.Series, months_back: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (absolute_change, pct_change) using the last value
    and the last value on or before (last_date - months_back).
    """
    series = series.dropna()
    if series.empty:
        return None, None

    last_date = series.index[-1]
    last_value = float(series.iloc[-1])
    ref_date = last_date - pd.DateOffset(months=months_back)
    ref_value = last_value_on_or_before(series, ref_date)
    if ref_value is None or ref_value == 0:
        return None, None

    abs_change = last_value - ref_value
    pct_change = (last_value / ref_value - 1.0) * 100.0
    return abs_change, pct_change


def value_on_or_before(series: pd.Series, target_date: pd.Timestamp) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    series = series.dropna()
    if series.empty:
        return None, None
    series = series.loc[series.index <= target_date]
    if series.empty:
        return None, None
    return series.index[-1], float(series.iloc[-1])


def format_change(abs_change: Optional[float], pct_change: Optional[float]) -> str:
    if abs_change is None or pct_change is None:
        return "n/a"
    sign = "+" if abs_change >= 0 else ""
    return f"{sign}{abs_change:,.2f} ({sign}{pct_change:.2f}%)"


def plot_spot_vs_basket(
    spot: pd.Series,
    basket: pd.Series,
) -> plt.Figure:
    plt.rcParams.update({
        "axes.facecolor": CHART_COLORS["panel"],
        "figure.facecolor": CHART_COLORS["bg"],
        "axes.edgecolor": CHART_COLORS["spine"],
        "axes.labelcolor": CHART_COLORS["text"],
        "xtick.color": CHART_COLORS["tick"],
        "ytick.color": CHART_COLORS["tick"],
        "grid.color": CHART_COLORS["grid"],
        "grid.alpha": 0.4,
    })
    fig, ax_top = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 4.6),
    )

    line1, = ax_top.plot(
        spot.index, spot.values,
        color=CHART_COLORS["spot"], linewidth=2.6,
        label="APT Scrap Spot (USD/MT)"
    )
    ax_top.set_ylabel("APT 88.5% Spot (USD/MT)", color=CHART_COLORS["spot"])
    ax_top.tick_params(axis="y", labelcolor=CHART_COLORS["spot"])
    ax_top.grid(True, alpha=0.25)
    ax_top.set_title(
        "Tungsten APT Spot vs Global APT Equity Basket (Spot-Anchored)",
        color=CHART_COLORS["text"]
    )

    ax_right = ax_top.twinx()
    line2, = ax_right.plot(
        basket.index, basket.values,
        color=CHART_COLORS["basket"], linewidth=3.2, linestyle="-",
        label="Tungsten APT Equity Basket (Start=100)"
    )
    ax_right.set_ylabel("Equity Basket Index (Start=100)", color=CHART_COLORS["basket"])
    ax_right.tick_params(axis="y", labelcolor=CHART_COLORS["basket"])

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax_top.legend(lines, labels, loc="upper left", frameon=True, labelcolor=CHART_COLORS["text"])

    fig.tight_layout()
    return fig


def plot_composite(
    composite_pct: pd.Series,
) -> plt.Figure:
    plt.rcParams.update({
        "axes.facecolor": CHART_COLORS["panel"],
        "figure.facecolor": CHART_COLORS["bg"],
        "axes.edgecolor": CHART_COLORS["spine"],
        "axes.labelcolor": CHART_COLORS["text"],
        "xtick.color": CHART_COLORS["tick"],
        "ytick.color": CHART_COLORS["tick"],
        "grid.color": CHART_COLORS["grid"],
        "grid.alpha": 0.4,
    })
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        composite_pct.index,
        composite_pct.values,
        color=CHART_COLORS["composite"],
        linewidth=2.6,
        label="Composite (% change from start)"
    )
    ax.axhline(0, color=CHART_COLORS["tick"], linewidth=1, alpha=0.6)
    ax.set_title("Composite Performance (Spot + Basket)", color=CHART_COLORS["text"])
    ax.set_ylabel("% Change", color=CHART_COLORS["text"])
    ax.set_xlabel("Date", color=CHART_COLORS["text"])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=True, labelcolor=CHART_COLORS["text"])
    fig.tight_layout()
    return fig


def change_badge(abs_change: Optional[float], pct_change: Optional[float]) -> str:
    if abs_change is None or pct_change is None:
        return '<span class="chg chg-na">n/a</span>'
    is_pos = abs_change >= 0
    arrow = "▲" if is_pos else "▼"
    sign = "+" if is_pos else ""
    text = f"{arrow} {sign}{abs_change:,.2f} ({sign}{pct_change:.2f}%)"
    cls = "chg chg-pos" if is_pos else "chg chg-neg"
    return f'<span class="{cls}">{text}</span>'


def pct_badge(pct_change: Optional[float]) -> str:
    if pct_change is None:
        return '<span class="chg chg-na">n/a</span>'
    is_pos = pct_change >= 0
    arrow = "▲" if is_pos else "▼"
    sign = "+" if is_pos else ""
    text = f"{arrow} {sign}{pct_change:.2f}%"
    cls = "chg chg-pos" if is_pos else "chg chg-neg"
    return f'<span class="{cls}">{text}</span>'


# =============================
# STREAMLIT APP
# =============================

st.set_page_config(
    page_title="Tungsten APT Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("Tungsten APT Dashboard")
st.caption("APT 88.5% spot (ScrapMonster) + global APT equity basket. Spot dates anchor the index.")

st.markdown(
    """
<style>
  .metric-card {
    padding: 12px 14px;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    background: linear-gradient(180deg, rgba(17,24,39,0.9), rgba(15,23,42,0.9));
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
  }
  .metric-title {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.7);
    margin-bottom: 6px;
  }
  .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    white-space: nowrap;
  }
  .metric-sub {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.85);
    margin-top: 6px;
  }
  .chg {
    font-weight: 700;
    white-space: nowrap;
  }
  .chg-pos { color: #22c55e; }
  .chg-neg { color: #ef4444; }
  .chg-na { color: rgba(255,255,255,0.5); }
</style>
""",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Controls")
    spot_series_choice = st.selectbox(
        "Spot series",
        options=["APT_AVG"] + SERIES_LABELS_IN_ORDER,
        index=0
    )
    w_spot = st.slider("Composite weight: spot", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    tickers_input = st.text_area(
        "Equity tickers (one per line)",
        value="\n".join(TUNGSTEN_TICKERS_DEFAULT),
        height=150
    )
    refresh = st.button("Refresh data")

    st.divider()
    st.subheader("Buy-price proxy (China benchmark)")

    use_buy_proxy = st.checkbox("Enable buy-price proxy", value=True)

    china_cny_per_tonne = st.number_input(
        "China APT benchmark (CNY/tonne)",
        min_value=0.0,
        value=950000.0,
        step=5000.0,
        help="Example: 950,000 CNY/tonne (APT 88.5% WO₃, ex-works China)."
    )

    usdcny = fetch_usdcny_rate_yf()
    if usdcny is None:
        usdcny = 7.20
        st.warning("USD/CNY rate unavailable from Yahoo; using fallback 7.20.")
    st.caption(f"USD/CNY (live from Yahoo): {usdcny:.4f}")

    incoterm_uplift_pct = st.slider(
        "Incoterm uplift (%)",
        min_value=-5.0,
        max_value=25.0,
        value=0.0,
        step=0.5,
        help="Adjust for terms (FOB/CIF), shipping, handling, etc. Positive = higher buy price."
    )

    buyer_premium_pct = st.slider(
        "Buyer premium (%)",
        min_value=-5.0,
        max_value=25.0,
        value=0.0,
        step=0.5,
        help="Your incremental premium/discount vs the China benchmark."
    )

    blend_to_spot_pct = st.slider(
        "Blend toward ScrapMonster spot (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0,
        help="0% = pure China-benchmark level-shifted series. 100% = pure ScrapMonster spot series."
    )


@st.cache_data(show_spinner=False)
def run_pipeline(
    spot_series_for_plot: str,
    tickers: List[str]
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DatetimeIndex, List[str], str]:
    html = fetch_html(APT_URL)
    latest_df, hist_df = parse_apt_page(html)
    _, hist_path = update_apt_files(latest_df, hist_df)

    spot_wide = load_spot_series(hist_path)
    if spot_series_for_plot not in spot_wide.columns:
        raise RuntimeError(f"Spot series '{spot_series_for_plot}' not available in spot data.")

    spot_series = spot_wide[spot_series_for_plot].dropna()
    if spot_series.empty:
        raise RuntimeError("Spot series is empty after dropna().")

    spot_dates = pd.DatetimeIndex(spot_series.index)
    start_date = spot_dates.min().date().isoformat()

    px = download_equities(tickers, start_date=start_date)
    missing = [t for t in tickers if t not in list(px.columns)]
    if px.shape[1] < 2:
        raise RuntimeError(
            f"Not enough equity tickers returned data ({px.shape[1]}). "
            f"Try adjusting tickers; current: {list(px.columns)}"
        )

    basket_on_spot = build_equal_weight_index_on_spot_dates(px, spot_dates)

    common_dates = spot_series.index.intersection(basket_on_spot.index)
    spot_final = spot_series.loc[common_dates]
    basket_final = basket_on_spot.loc[common_dates]
    composite_pct = build_composite_pct(spot_final, basket_final, w_spot=w_spot)

    last_updated = now_utc_iso()
    return spot_final, basket_final, composite_pct, px, spot_dates, missing, last_updated


if refresh:
    run_pipeline.clear()

tickers = [t.strip() for t in tickers_input.splitlines() if t.strip()]

with st.spinner("Loading tungsten data..."):
    try:
        spot_final, basket_final, composite_pct, px, spot_dates, missing_tickers, last_updated = run_pipeline(
            spot_series_choice, tickers
        )
    except Exception as e:
        st.error(str(e))
        st.stop()


# -----------------------------
# Optional: China benchmark buy-price proxy
# -----------------------------
buy_price_series = pd.Series(dtype="float64")
buy_price_today = float("nan")
spot_for_model = spot_final

if "use_buy_proxy" in globals() and use_buy_proxy:
    try:
        buy_price_series, buy_price_today = modeled_buy_price_series_from_china_benchmark(
            spot_series_usd_mt=spot_final,
            china_cny_per_tonne=china_cny_per_tonne,
            usdcny=usdcny,
            incoterm_uplift_pct=incoterm_uplift_pct,
            buyer_premium_pct=buyer_premium_pct,
            blend_with_spot=(blend_to_spot_pct / 100.0),
        )
        if not buy_price_series.empty:
            spot_for_model = buy_price_series
    except Exception:
        buy_price_series = pd.Series(dtype="float64")
        buy_price_today = float("nan")
        spot_for_model = spot_final



tab1, tab2, tab3 = st.tabs(["Dashboard", "Tungsten APT Equity Index", "Modeled Price"])

with tab1:
    if missing_tickers:
        st.warning(f"Tickers with no equity data (spot-anchored window): {', '.join(missing_tickers)}")
    latest_spot = float(spot_final.iloc[-1])
    latest_basket = float(basket_final.iloc[-1])

    basket_wow_change = compute_change(basket_final, 7)
    basket_mom_change = compute_change(basket_final, 30)
    basket_qoq_change = compute_change(basket_final, 90)
    composite_index = 100.0 * (1.0 + (composite_pct / 100.0))
    composite_wow_change = compute_change(composite_index, 7)
    composite_mom_change = compute_change(composite_index, 30)
    composite_qoq_change = compute_change(composite_index, 90)


    if "use_buy_proxy" in globals() and use_buy_proxy and pd.notna(buy_price_today):
        buy_wow_change = compute_change(buy_price_series, 7)
        buy_mom_change = compute_change(buy_price_series, 30)
        buy_qoq_change = compute_change(buy_price_series, 90)
        col_bp1, col_bp2, col_bp3, col_bp4 = st.columns([1.4, 1, 1, 1])
        col_bp1.markdown(
            f'''
<div class="metric-card" style="margin-top: 10px;">
  <div class="metric-title">Modeled Buy Price (USD/MT)</div>
  <div class="metric-value">{buy_price_today:,.0f}</div>
  <div class="metric-sub">From China benchmark: {china_cny_per_tonne:,.0f} CNY/t @ {usdcny:.4f} USD/CNY</div>
</div>
''',
            unsafe_allow_html=True
        )
        col_bp2.markdown(
            f'''
<div class="metric-card" style="margin-top: 10px;">
  <div class="metric-title">WoW</div>
  <div class="metric-sub">{change_badge(*buy_wow_change)}</div>
</div>
''',
            unsafe_allow_html=True
        )
        col_bp3.markdown(
            f'''
<div class="metric-card" style="margin-top: 10px;">
  <div class="metric-title">MoM</div>
  <div class="metric-sub">{change_badge(*buy_mom_change)}</div>
</div>
''',
            unsafe_allow_html=True
        )
        col_bp4.markdown(
            f'''
<div class="metric-card" style="margin-top: 10px;">
  <div class="metric-title">QoQ</div>
  <div class="metric-sub">{change_badge(*buy_qoq_change)}</div>
</div>
''',
            unsafe_allow_html=True
        )
    # Equity-implied price (basket-only, monthly-aligned) for quick glance on dashboard
    equity_implied = None
    equity_implied_series = pd.Series(dtype="float64")
    try:
        spot_series_for_implied = spot_for_model.copy()
        X_implied = basket_final.to_frame("Basket")
        if len(spot_series_for_implied) > 0 and len(X_implied) > 0:
            spot_m = to_monthly_last(spot_series_for_implied)
            X_m = to_monthly_last_df(X_implied)
            end_dt = spot_m.index.max()
            start_dt = end_dt - pd.DateOffset(years=3)
            spot_w = spot_m.loc[spot_m.index >= start_dt]
            X_w = X_m.loc[X_m.index >= start_dt]
            coef_i, intercept_i, modeled_i, X_mean_i, X_std_i = fit_linear_model(X_w, spot_w)
            if modeled_i is not None and len(modeled_i) > 0:
                equity_implied_series = modeled_i.copy()
                X_latest_i = X_m.loc[X_m.index <= end_dt].iloc[-1]
                Xz_latest_i = (X_latest_i - X_mean_i) / X_std_i.replace(0, np.nan)
                Xz_latest_i = Xz_latest_i.fillna(0.0)
                equity_implied = float(intercept_i + np.dot(coef_i, Xz_latest_i.values))
    except Exception:
        equity_implied = None

    if equity_implied is not None and not np.isnan(equity_implied):
        # Build a daily implied series from the monthly-fitted coefficients
        implied_daily = pd.Series(dtype="float64")
        basket_daily_all = build_equal_weight_index_daily(px)
        if (not basket_daily_all.empty) and (coef_i is not None) and (len(coef_i) > 0):
            X_daily = basket_daily_all.to_frame("Basket")
            Xz_daily = (X_daily - X_mean_i) / X_std_i.replace(0, np.nan)
            Xz_daily = Xz_daily.fillna(0.0)
            implied_daily = (Xz_daily @ coef_i) + intercept_i
            implied_daily.name = "Equity_Implied_APT_Daily"

        series_for_change = implied_daily if not implied_daily.empty else equity_implied_series
        series_label = "Daily implied" if not implied_daily.empty else "Monthly implied"
        eq_wow_change = compute_change(series_for_change, 7)
        eq_mom_change = compute_months_change(series_for_change, 1)
        eq_qoq_change = compute_months_change(series_for_change, 3)
        col_eq1, col_eq2, col_eq3, col_eq4 = st.columns([1.4, 1, 1, 1])
        col_eq1.markdown(
            f'''
<div class="metric-card" style="margin-top: 10px;">
  <div class="metric-title">Equity-Implied APT (USD/MT)</div>
  <div class="metric-value">{equity_implied:,.0f}</div>
</div>
''',
            unsafe_allow_html=True
        )
        col_eq2.markdown(
            f'''
<div class="metric-card" style="margin-top: 10px;">
  <div class="metric-title">WoW</div>
  <div class="metric-sub">{change_badge(*eq_wow_change)}</div>
</div>
''',
            unsafe_allow_html=True
        )
        col_eq3.markdown(
            f'''
<div class="metric-card" style="margin-top: 10px;">
  <div class="metric-title">MoM</div>
  <div class="metric-sub">{change_badge(*eq_mom_change)}</div>
</div>
''',
            unsafe_allow_html=True
        )
        col_eq4.markdown(
            f'''
<div class="metric-card" style="margin-top: 10px;">
  <div class="metric-title">QoQ</div>
  <div class="metric-sub">{change_badge(*eq_qoq_change)}</div>
</div>
''',
            unsafe_allow_html=True
        )

        with st.expander("Equity-Implied Calc Details"):
            last_dt, last_val = value_on_or_before(series_for_change, series_for_change.index.max())
            wow_dt, wow_val = value_on_or_before(series_for_change, series_for_change.index.max() - pd.Timedelta(days=7))
            mom_dt, mom_val = value_on_or_before(series_for_change, series_for_change.index.max() - pd.DateOffset(months=1))
            qoq_dt, qoq_val = value_on_or_before(series_for_change, series_for_change.index.max() - pd.DateOffset(months=3))
            st.caption(f"Series used: {series_label}")
            st.caption(f"Latest: {last_dt.date() if last_dt is not None else 'n/a'} | {last_val:,.2f}" if last_dt is not None else "Latest: n/a")
            st.caption(f"WoW ref: {wow_dt.date() if wow_dt is not None else 'n/a'} | {wow_val:,.2f}" if wow_dt is not None else "WoW ref: n/a")
            st.caption(f"MoM ref: {mom_dt.date() if mom_dt is not None else 'n/a'} | {mom_val:,.2f}" if mom_dt is not None else "MoM ref: n/a")
            st.caption(f"QoQ ref: {qoq_dt.date() if qoq_dt is not None else 'n/a'} | {qoq_val:,.2f}" if qoq_dt is not None else "QoQ ref: n/a")

    col_sp1 = st.columns([1])[0]
    col_sp1.markdown(
        f"""
<div class="metric-card" style="margin-top: 10px;">
  <div class="metric-title">Spot Scrap Price (USD/MT)</div>
  <div class="metric-value">{latest_spot:,.2f}</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.divider()

    col5, col6, col7, col8 = st.columns([1.4, 1, 1, 1])
    col5.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">Equity Basket (Start=100)</div>
  <div class="metric-value">{latest_basket:,.2f}</div>
</div>
""",
        unsafe_allow_html=True
    )
    col6.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">WoW</div>
  <div class="metric-sub">{change_badge(*basket_wow_change)}</div>
</div>
""",
        unsafe_allow_html=True
    )
    col7.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">MoM</div>
  <div class="metric-sub">{change_badge(*basket_mom_change)}</div>
</div>
""",
        unsafe_allow_html=True
    )
    col8.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">QoQ</div>
  <div class="metric-sub">{change_badge(*basket_qoq_change)}</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.caption(f"Data refreshed: {last_updated} (UTC)")

    fig_spot = plot_spot_vs_basket(spot_final, basket_final)
    st.pyplot(fig_spot, use_container_width=True)

    st.divider()

    col9, col10, col11, col12 = st.columns([1.4, 1, 1, 1])
    col9.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">Composite Index (Start=100)</div>
  <div class="metric-value">{composite_index.iloc[-1]:,.2f}</div>
</div>
""",
        unsafe_allow_html=True
    )
    col10.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">WoW</div>
  <div class="metric-sub">{change_badge(*composite_wow_change)}</div>
</div>
""",
        unsafe_allow_html=True
    )
    col11.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">MoM</div>
  <div class="metric-sub">{change_badge(*composite_mom_change)}</div>
</div>
""",
        unsafe_allow_html=True
    )
    col12.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">QoQ</div>
  <div class="metric-sub">{change_badge(*composite_qoq_change)}</div>
</div>
""",
        unsafe_allow_html=True
    )

    fig_comp = plot_composite(composite_pct)
    st.pyplot(fig_comp, use_container_width=True)

    st.subheader("Spot + Basket (Spot-Anchored)")
    out_df = pd.DataFrame({
        "APT_spot_usd_mt": spot_final,
        "Basket_index_start_100": basket_final,
        "Composite_pct_change": composite_pct
    })
    st.dataframe(out_df.tail(250), use_container_width=True)

with tab2:
    st.subheader("Holdings Performance (APT, Normalized to 100)")
    px_5y = download_equities_5y(tickers)
    if px_5y.empty:
        st.info("Holdings data is not available for the selected tickers.")
    else:
        missing_5y = [t for t in tickers if t not in list(px_5y.columns)]
        if missing_5y:
            st.warning(f"Tickers with no 5Y equity data: {', '.join(missing_5y)}")
        px_5y = px_5y.sort_index()
        px_5y_ffill = px_5y.ffill()
        end_date = px_5y.index.max()
        ytd_start = pd.Timestamp(year=end_date.year, month=1, day=1)
        window_options = {
            "1W": end_date - pd.DateOffset(weeks=1),
            "1M": end_date - pd.DateOffset(months=1),
            "YTD": ytd_start,
            "1Y": end_date - pd.DateOffset(years=1),
            "3Y": end_date - pd.DateOffset(years=3),
            "5Y": end_date - pd.DateOffset(years=5),
        }
        window_choice = st.radio(
            "Timeframe",
            options=["1W", "1M", "YTD", "1Y", "3Y", "5Y"],
            horizontal=True
        )
        start_date = window_options[window_choice]
        px_window = px_5y_ffill.loc[px_5y_ffill.index >= start_date]
        if px_window.empty:
            st.info("No holdings data available for the selected timeframe.")
            st.stop()

        px_window = px_window.dropna(axis=1, how="all")
        holdings_norm = normalize_prices_window(px_window)
        basket_daily = build_equal_weight_index_daily(px_window)
        basket_total_return = (basket_daily.iloc[-1] / basket_daily.iloc[0] - 1.0) * 100.0
        st.markdown(
            f"""
<div class="metric-card" style="max-width: 360px; margin-bottom: 12px;">
  <div class="metric-title">Basket Total Return ({window_choice})</div>
  <div class="metric-sub">{pct_badge(basket_total_return)}</div>
</div>
""",
            unsafe_allow_html=True
        )

        use_intraday = st.checkbox("Use intraday last price for DoD%", value=True)
        if use_intraday:
            last_px, prev_close = fetch_intraday_last_and_prev_close(list(px_window.columns))
            valid = (last_px.notna()) & (prev_close.notna()) & (prev_close != 0)
            if valid.any():
                rets = (last_px[valid] / prev_close[valid]) - 1.0
                basket_dod_pct = float(rets.mean() * 100.0)
                basket_dod_change = (np.nan, basket_dod_pct)
            else:
                basket_dod_change = compute_change(basket_daily, 1)
        else:
            basket_dod_change = compute_change(basket_daily, 1)
        basket_wow_change = compute_change(basket_daily, 7)
        basket_mom_change = compute_change(basket_daily, 30)
        col_a, col_b, col_c = st.columns(3)
        col_a.markdown(
            f"""
<div class="metric-card">
  <div class="metric-title">DoD%</div>
  <div class="metric-sub">{pct_badge(basket_dod_change[1])}</div>
</div>
""",
            unsafe_allow_html=True
        )
        col_b.markdown(
            f"""
<div class="metric-card">
  <div class="metric-title">WoW%</div>
  <div class="metric-sub">{pct_badge(basket_wow_change[1])}</div>
</div>
""",
            unsafe_allow_html=True
        )
        col_c.markdown(
            f"""
<div class="metric-card">
  <div class="metric-title">MoM%</div>
  <div class="metric-sub">{pct_badge(basket_mom_change[1])}</div>
</div>
""",
            unsafe_allow_html=True
        )

        total_returns = (px_window.iloc[-1] / px_window.iloc[0] - 1.0) * 100.0
        total_returns = total_returns.sort_values(ascending=False).to_frame("Total_Return_%")
        fig2, ax = plt.subplots(figsize=(11, 6))
        ax.set_facecolor(CHART_COLORS["panel"])
        for col in holdings_norm.columns:
            ax.plot(
                holdings_norm.index,
                holdings_norm[col],
                linewidth=1.4,
                alpha=0.75,
                label=ticker_label(col)
            )

        basket_plot = basket_daily.reindex(holdings_norm.index).dropna()
        plot_df = holdings_norm.copy()
        plot_df["Basket Index"] = basket_plot
        plot_df = plot_df.rename(columns={c: ticker_label(c) for c in holdings_norm.columns})

        fig = go.Figure()
        for col in plot_df.columns:
            is_basket = col == "Basket Index"
            fig.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[col],
                    mode="lines",
                    name=col,
                    line=dict(
                        width=3.6 if is_basket else 1.6,
                        color="#ffffff" if is_basket else None,
                    ),
                )
            )

        fig.update_layout(
            template="plotly_dark",
            title="Holdings + APT Basket Performance (5Y Equity Window, Start=100)",
            xaxis_title="Date",
            yaxis_title="Index (Start=100)",
            hovermode="closest",
            legend=dict(orientation="v"),
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Holdings Total Return")
        st.dataframe(total_returns, use_container_width=True)

with tab3:
    st.subheader("Modeled Tungsten APT Price (Equities → Spot)")

    model_basis = st.radio(
        "Model inputs",
        options=["Basket only", "All holdings"],
        horizontal=True
    )
    window_choice = st.selectbox("Training window", options=["YTD", "1Y", "3Y", "5Y", "All"], index=2)
    use_monthly = st.checkbox("Align to monthly (last value)", value=True, help="Recommended for monthly/lagged spot series.")

    spot_series = spot_for_model.copy()
    st.caption(
        "Model target uses: "
        + (
            "Buy-price proxy (China benchmark)"
            if (("use_buy_proxy" in globals()) and use_buy_proxy and (not buy_price_series.empty))
            else "ScrapMonster spot"
        )
    )
    px_on_spot = align_equities_to_spot(px, spot_series.index)
    px_on_spot = px_on_spot.dropna(axis=1, how="all")

    if model_basis == "Basket only":
        X = basket_final.to_frame("Basket")
    else:
        X = normalize_from_start(px_on_spot)

    if use_monthly:
        spot_series = to_monthly_last(spot_series)
        X = to_monthly_last_df(X)

    end_date = spot_series.index.max()
    if window_choice == "YTD":
        start_date = pd.Timestamp(year=end_date.year, month=1, day=1)
    elif window_choice == "1Y":
        start_date = end_date - pd.DateOffset(years=1)
    elif window_choice == "3Y":
        start_date = end_date - pd.DateOffset(years=3)
    elif window_choice == "5Y":
        start_date = end_date - pd.DateOffset(years=5)
    else:
        start_date = spot_series.index.min()

    spot_window = spot_series.loc[spot_series.index >= start_date]
    X_window = X.loc[X.index >= start_date]

    coef, intercept, modeled, X_mean, X_std = fit_linear_model(X_window, spot_window)
    if modeled.empty:
        st.info("Not enough overlapping data to fit a model.")
        st.stop()

    modeled_full = modeled.reindex(spot_series.index).ffill()
    r2, rmse, corr = model_metrics(spot_window, modeled)

    # Implied current spot from latest equity inputs
    X_latest = X.loc[X.index <= end_date].iloc[-1]
    Xz_latest = (X_latest - X_mean) / X_std.replace(0, np.nan)
    Xz_latest = Xz_latest.fillna(0.0)
    implied_current = float(intercept + np.dot(coef, Xz_latest.values))

    col_a, col_b, col_c = st.columns(3)
    col_a.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">Correlation</div>
  <div class="metric-sub">{pct_badge(corr * 100 if pd.notna(corr) else None)}</div>
</div>
""",
        unsafe_allow_html=True
    )
    col_b.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">R²</div>
  <div class="metric-sub">{pct_badge(r2 * 100 if pd.notna(r2) else None)}</div>
</div>
""",
        unsafe_allow_html=True
    )
    col_c.markdown(
        f"""
<div class="metric-card">
  <div class="metric-title">RMSE (USD/MT)</div>
  <div class="metric-sub">{'n/a' if pd.isna(rmse) else f'{rmse:,.0f}'}</div>
</div>
""",
        unsafe_allow_html=True
    )

    last_spot = float(spot_series.iloc[-1])
    premium_pct = ((implied_current / last_spot) - 1.0) * 100.0 if last_spot != 0 else None
    st.markdown(
        f"""
<div class="metric-card" style="max-width: 420px; margin-bottom: 12px;">
  <div class="metric-title">Implied Current Spot (USD/MT)</div>
  <div class="metric-value">{implied_current:,.0f}</div>
  <div class="metric-sub">Premium/Discount vs last spot: {pct_badge(premium_pct)}</div>
</div>
""",
        unsafe_allow_html=True
    )

    fig_model, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_facecolor(CHART_COLORS["panel"])
    ax.plot(spot_series.index, spot_series.values, color=CHART_COLORS["spot"], linewidth=2.4, label="Actual Spot")
    ax.plot(modeled_full.index, modeled_full.values, color=CHART_COLORS["composite"], linewidth=2.4, label="Modeled Spot")
    ax.set_title("Modeled Tungsten APT Spot from Equity Signals", color=CHART_COLORS["text"])
    ax.set_ylabel("USD/MT", color=CHART_COLORS["text"])
    ax.set_xlabel("Date", color=CHART_COLORS["text"])
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CHART_COLORS["spine"])
    ax.spines["bottom"].set_color(CHART_COLORS["spine"])
    ax.tick_params(axis="both", colors=CHART_COLORS["tick"])
    ax.legend(loc="upper left", frameon=True, labelcolor=CHART_COLORS["text"])
    st.pyplot(fig_model, use_container_width=True)

    if model_basis == "All holdings" and coef.size:
        coef_df = pd.DataFrame({
            "Ticker": X_window.columns,
            "Company": [ticker_label(t) for t in X_window.columns],
            "Coefficient": coef,
        }).sort_values("Coefficient", ascending=False)
        st.subheader("Implied Weights (Standardized)")
        st.dataframe(coef_df, use_container_width=True)
