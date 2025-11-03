import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import os
import requests
import re
from dotenv import load_dotenv
from SmartApi.smartConnect import SmartConnect
import pyotp
from typing import Optional
import io, zipfile  # NEW: for robust mobile file reading

# ================== App setup ==================
st.set_page_config(page_title="Chartink Visualizer • Axis Zoom + Limited Window", layout="wide")
st.title("Chartink Screener Visualizer")
st.caption("Axis-drag zoom: drag X-axis to zoom horizontally, drag Y-axis to zoom vertically. Only a compact window around the signal is rendered.")

# ================== Credentials ==================
load_dotenv()
API_KEY = os.getenv("SMARTAPI_API_KEY")
CLIENT_CODE = os.getenv("SMARTAPI_CLIENT_CODE")
PIN = os.getenv("SMARTAPI_PIN")
TOTP_SECRET = os.getenv("SMARTAPI_TOTP_SECRET")

IST = pytz.timezone("Asia/Kolkata")

@st.cache_data(show_spinner=False)
def load_token_map():
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    data = requests.get(url, timeout=15).json()
    return {e["symbol"]: e["token"] for e in data if e["exch_seg"] == "NSE" and e["instrumenttype"] == ""}

def smartapi_login():
    smartApi = SmartConnect(api_key=API_KEY)
    totp = pyotp.TOTP(TOTP_SECRET).now()
    data = smartApi.generateSession(CLIENT_CODE, PIN, totp)
    if not data or not data.get("status") or not data.get("data") or not data["data"].get("jwtToken"):
        st.error(f"SmartAPI login failed. Raw response: {data}")
        raise RuntimeError("SmartAPI login failed.")
    return smartApi

# ================== Timeframes ==================
TIMEFRAME = {
    "1m":  {"native": "ONE_MINUTE",    "base": "ONE_MINUTE",   "rule": "1min",  "intraday": True,  "bars_per_day": 375, "secs": 60},
    "3m":  {"native": "THREE_MINUTE",  "base": "ONE_MINUTE",   "rule": "3min",  "intraday": True,  "bars_per_day": 125, "secs": 180},
    "5m":  {"native": "FIVE_MINUTE",   "base": "FIVE_MINUTE",  "rule": "5min",  "intraday": True,  "bars_per_day": 75,  "secs": 300},
    "10m": {"native": "TEN_MINUTE",    "base": "FIVE_MINUTE",  "rule": "10min", "intraday": True,  "bars_per_day": 37,  "secs": 600},
    "15m": {"native": "FIFTEEN_MINUTE","base": "FIVE_MINUTE",  "rule": "15min", "intraday": True,  "bars_per_day": 25,  "secs": 900},
    "30m": {"native": "THIRTY_MINUTE", "base": "FIVE_MINUTE",  "rule": "30min", "intraday": True,  "bars_per_day": 12,  "secs": 1800},
    "1h":  {"native": "ONE_HOUR",      "base": "FIVE_MINUTE",  "rule": "1h",    "intraday": True,  "bars_per_day": 6,   "secs": 3600},
    "1d":  {"native": "ONE_DAY",       "base": "ONE_DAY",      "rule": None,    "intraday": False, "bars_per_day": 1,   "secs": 86400},
}
TF_CHOICES = list(TIMEFRAME.keys())

def _parse_angel_timestamp(series: pd.Series) -> pd.DatetimeIndex:
    s = pd.Series(series)
    ts = pd.to_datetime(s, errors="coerce", utc=False)
    if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
        ts = ts.dt.tz_convert(IST).dt.tz_localize(None)
    return pd.DatetimeIndex(ts)

def _anchor_bar_start(ts: datetime, frame_delta: pd.Timedelta) -> datetime:
    if frame_delta >= pd.Timedelta("1d"):
        return datetime(ts.year, ts.month, ts.day)
    day_start = ts.replace(hour=9, minute=15, second=0, microsecond=0)
    if ts <= day_start:
        return day_start
    elapsed = ts - day_start
    steps = int(elapsed.total_seconds() // frame_delta.total_seconds())
    return day_start + steps * frame_delta

def _drop_bars_crossing_close(df: pd.DataFrame, frame_delta: pd.Timedelta) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df.index
    close_times = pd.to_datetime([datetime(d.year, d.month, d.day, 15, 30) for d in idx])
    ends = idx + frame_delta
    within = ends <= close_times
    last_bar = df.groupby(idx.normalize(), sort=False).tail(1).index
    return df.loc[within | idx.isin(last_bar)]

def parse_screen_dt(val: str) -> Optional[datetime]:
    s = str(val).strip()
    if not s: return None
    ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.isna(ts):
        for fmt in ("%d-%m-%Y %H:%M", "%d-%m-%Y %I:%M %p", "%Y-%m-%d %H:%M:%S"):
            try: return datetime.strptime(s, fmt)
            except: continue
        return None
    has_time = bool(re.search(r"\d{1,2}:\d{2}", s))
    dt = ts.to_pydatetime()
    if not has_time: dt = dt.replace(hour=9, minute=15, second=0, microsecond=0)
    return dt

def _fetch_bars_windowed(
    smartApi,
    token: str,
    signal_dt_ist: datetime,
    tf_key: str,
    required_bars: int
):
    """
    Fetch only enough history around the signal to render required_bars (+ small buffer).
    """
    cfg = TIMEFRAME.get(tf_key, TIMEFRAME["15m"])
    intraday = cfg["intraday"]
    bpd = cfg["bars_per_day"]
    rule = cfg["rule"]
    base_for_resample = cfg["base"]
    native = cfg["native"]

    # compute days needed
    buffer_bars = 10
    bars = required_bars + buffer_bars
    if intraday:
        days = max(3, int(np.ceil(bars / max(1, bpd))) + 2)
    else:
        days = max(5, bars + 2)

    start_dt = signal_dt_ist - timedelta(days=days)
    end_dt = signal_dt_ist + timedelta(days=days)

    params = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": (base_for_resample if intraday else native),
        "fromdate": start_dt.strftime("%Y-%m-%d %H:%M"),
        "todate": end_dt.strftime("%Y-%m-%d %H:%M"),
    }
    try:
        historic_data = smartApi.getCandleData(params)
    except Exception as e:
        st.warning(f"Fetch error for token={token}: {e}")
        return None, cfg

    if not historic_data or not historic_data.get("data"):
        return None, cfg

    candles = historic_data["data"]
    df = pd.DataFrame(candles, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
    idx = _parse_angel_timestamp(df["timestamp"])
    df = df.drop(columns=["timestamp"])
    df.index = idx
    df = df.dropna(subset=["Open","High","Low","Close"])
    if df.empty: return None, cfg

    if intraday:
        try:
            df = df.between_time("09:15", "15:30", inclusive="both")
        except TypeError:
            df = df.between_time("09:15", "15:30")
        if df.empty: return None, cfg

        ohlc = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
        df = df.resample(
            rule or "1h",
            origin="start_day",
            offset="15min",
            label="left",
            closed="left"
        ).agg(ohlc).dropna()
        if df.empty: return None, cfg
        frame_delta = pd.to_timedelta(rule or "1h")
        df = _drop_bars_crossing_close(df, frame_delta)
        end_of_day = pd.to_datetime([datetime(d.year, d.month, d.day, 15, 30) for d in df.index])
        df = df[df.index <= end_of_day]

    return df, cfg

# ================== Indicators (no-partial) ==================
def wma(series: pd.Series, length: int) -> pd.Series:
    L = int(max(1, length))
    weights = np.arange(1, L + 1, dtype=float)
    return series.astype(float).rolling(L, min_periods=L).apply(
        lambda x: float(np.dot(x, weights[-len(x):]) / weights[-len(x):].sum()), raw=True
    )

def hma_no_partial(series: pd.Series, length: int) -> pd.Series:
    L = int(max(1, length))
    half = int(np.ceil(L / 2.0))
    sqrtL = int(np.ceil(np.sqrt(L)))
    wma_half = wma(series, half)
    wma_full = wma(series, L)
    diff = 2.0 * wma_half - wma_full
    return wma(diff, sqrtL)  # min_periods handled in wma

def ema_no_partial(series: pd.Series, length: int) -> pd.Series:
    L = int(max(1, length))
    ema = series.astype(float).ewm(alpha=2.0/(L+1.0), adjust=False).mean()
    if len(ema) >= L:
        ema.iloc[:L-1] = np.nan
    return ema

# ================== Mobile-friendly reader (Android/iOS) ==================
def read_uploaded_df(uploaded_file) -> pd.DataFrame:
    """
    Robustly read CSV/TXT/XLSX/XLS or a ZIP (with CSV) from desktop or mobile pickers.
    Works around Android MIME quirks by sniffing both extension and MIME.
    """
    raw = uploaded_file.read()
    name = (uploaded_file.name or "").lower()
    mime = (uploaded_file.type or "").lower()

    def try_csv(buf: bytes) -> pd.DataFrame:
        bio = io.BytesIO(buf)
        return pd.read_csv(bio, engine="python", encoding="utf-8", on_bad_lines="skip")

    def try_excel(buf: bytes) -> pd.DataFrame:
        bio = io.BytesIO(buf)
        return pd.read_excel(bio)  # requires openpyxl for .xlsx

    # Extension-based
    if name.endswith((".csv", ".txt", ".csv.gz")):
        return try_csv(raw)
    if name.endswith((".xlsx", ".xls")):
        return try_excel(raw)
    if name.endswith(".zip"):
        zf = zipfile.ZipFile(io.BytesIO(raw))
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError("Zip does not contain a CSV file.")
        with zf.open(csv_members[0]) as f:
            return pd.read_csv(f, engine="python", encoding="utf-8", on_bad_lines="skip")

    # MIME-based (mobile often uses vendor-specific types)
    if mime in {"text/csv", "application/csv", "text/plain"}:
        return try_csv(raw)
    if mime in {"application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}:
        return try_excel(raw)
    if mime == "application/zip":
        zf = zipfile.ZipFile(io.BytesIO(raw))
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError("Zip does not contain a CSV file.")
        with zf.open(csv_members[0]) as f:
            return pd.read_csv(f, engine="python", encoding="utf-8", on_bad_lines="skip")

    # Last resort: try CSV then Excel
    try:
        return try_csv(raw)
    except Exception:
        pass
    try:
        return try_excel(raw)
    except Exception:
        pass

    raise ValueError("Unsupported or unreadable file. Please upload CSV/TXT/XLSX/XLS, or a ZIP containing a CSV.")

# ================== Sidebar ==================
with st.sidebar:
    st.header("Input")

    # NEW: Mobile-friendly toggle + flexible file types
    mobile_friendly = st.toggle(
        "Mobile-friendly file picker (Android/iOS)",
        value=True,
        help="If files look greyed out on Android, enable this to select any file. The app will auto-detect CSV/XLSX/ZIP."
    )
    accepted_types = None if mobile_friendly else ["csv"]  # None = allow all; better for mobile pickers

    uploaded = st.file_uploader(
        "Upload screener file",
        type=accepted_types,
        accept_multiple_files=False
    )

    tf = st.selectbox("Timeframe", TF_CHOICES, index=TF_CHOICES.index("15m"))

    st.subheader("CSV column mapping")
    date_col = st.text_input("Screening Date/Time column", value="date")
    symbol_col = st.text_input("Symbol column", value="symbol")
    sector_col = st.text_input("Sector column (optional)", value="")

    st.subheader("Window around signal")
    auto_fit = st.toggle("Auto-fit candles to chart width", value=True,
                         help="Auto computes how many candles fit horizontally at your target pixels per candle.")
    target_px_per_candle = st.slider("Target pixels per candle", 6, 20, 12, 1)
    approx_chart_width_px = st.slider("Approx chart width (px)", 800, 1800, 1100, 50,
                                      help="Used only for auto-fit. Change if your container is wider/narrower.")
    n_before = st.slider("Candles before signal (manual mode)", 5, 120, 20, 1)
    n_after = st.slider("Candles after signal (manual mode)", 5, 120, 20, 1)

    st.subheader("Chart behavior")
    gapless = st.toggle("Gapless (remove non-trading gaps visually)", value=True,
                        help="Shows candles back-to-back using a linear index; tick labels still show time.")
    chart_h = st.slider("Chart height (px)", 360, 900, 560, 10)

    st.subheader("Pagination")
    initial_charts = st.number_input("Initial charts to load", min_value=1, max_value=100, value=12, step=1)
    charts_per_click = st.number_input("Charts per click (Load more)", min_value=1, max_value=100, value=10, step=1)

    st.caption("Axis-drag zoom: drag on the X-axis to zoom horizontally; drag on the Y-axis to zoom vertically. Double-click to reset.")

# ================== Main ==================
token_map = load_token_map()

if uploaded:
    # Use robust reader (works on Android/iOS/desktop). Make a fresh buffer per read:
    try:
        # Important: we already consumed the file bytes in read_uploaded_df.
        # For later references (like uploaded.name) we can still read .name, but not the stream.
        src_df = read_uploaded_df(uploaded)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()

    # Validate mapping
    for c in [date_col, symbol_col]:
        if c not in src_df.columns:
            st.error(f"Column '{c}' not found in CSV.")
            st.stop()
    if sector_col and (sector_col not in src_df.columns):
        st.warning(f"Sector column '{sector_col}' not found; sector will be blank.")
        sector_col_used = ""
    else:
        sector_col_used = sector_col

    df_screen = src_df.copy()
    df_screen["__screen_dt"] = df_screen[date_col].apply(parse_screen_dt)
    df_screen = df_screen.dropna(subset=["__screen_dt"]).sort_values("__screen_dt", ascending=False).reset_index(drop=True)

    # Login
    try:
        smartApi = smartapi_login()
    except Exception:
        st.stop()

    # Pagination state reset when input scope changes
    # Note: uploaded.name is safe to read even if file bytes are consumed
    uploaded_name = getattr(uploaded, "name", "upload")
    scope_key = f"{uploaded_name}|{len(df_screen)}|{tf}|{gapless}|{auto_fit}|{target_px_per_candle}|{approx_chart_width_px}|{n_before}|{n_after}"
    if st.session_state.get("scope_key") != scope_key:
        st.session_state.scope_key = scope_key
        st.session_state.display_limit = int(initial_charts)

    display_limit = int(st.session_state.display_limit)

    charts_rendered = 0
    EMA_LEN = 200
    HMA_LEN = 60
    HMA_WARMUP = HMA_LEN + int(np.ceil(np.sqrt(HMA_LEN))) + 5
    INDICATOR_WARMUP = max(EMA_LEN, HMA_WARMUP)

    total_rows = len(df_screen)
    for idx in range(total_rows):
        if charts_rendered >= display_limit:
            break

        row = df_screen.iloc[idx]
        signal_dt_naive = row["__screen_dt"]
        symbol = str(row[symbol_col]).strip().upper()
        symbol_eq = f"{symbol}-EQ"
        token = token_map.get(symbol_eq)
        if not token or pd.isna(signal_dt_naive):
            continue

        # Determine window size
        if auto_fit:
            desired = max(10, int(approx_chart_width_px // max(6, target_px_per_candle)))
            n_before_eff = desired // 2
            n_after_eff = desired - n_before_eff
        else:
            n_before_eff = int(n_before)
            n_after_eff = int(n_after)

        # Fetch minimal data around signal (enough for indicators + view)
        signal_dt_ist = IST.localize(signal_dt_naive)
        required_bars_fetch = (n_before_eff + n_after_eff) + INDICATOR_WARMUP + 10
        df_bars, cfg = _fetch_bars_windowed(
            smartApi, token, signal_dt_ist, tf_key=tf, required_bars=required_bars_fetch
        )
        if df_bars is None or df_bars.empty:
            continue

        # Locate signal candle start aligned to TF
        rule = cfg["rule"]
        frame_delta = (pd.Timedelta("1d") if rule is None else pd.to_timedelta(rule))
        signal_dt_naive_ist = signal_dt_ist.replace(tzinfo=None)
        candle_start = _anchor_bar_start(signal_dt_naive_ist, frame_delta)

        if candle_start in df_bars.index:
            sig_pos_full = df_bars.index.get_loc(candle_start)
        else:
            loc = df_bars.index.get_indexer([candle_start], method="pad")
            sig_pos_full = int(loc[0]) if loc[0] != -1 else 0

        # Plot/view window
        start_pos_plot = max(0, sig_pos_full - n_before_eff)
        end_pos_plot = min(len(df_bars), sig_pos_full + n_after_eff + 1)

        # Compute slice for indicators (extended)
        start_pos_compute = max(0, start_pos_plot - INDICATOR_WARMUP)
        df_compute = df_bars.iloc[start_pos_compute:end_pos_plot].copy()
        df_view = df_bars.iloc[start_pos_plot:end_pos_plot].copy()
        if df_view.empty:
            continue

        # Indicators (no partial)
        src = df_compute["Close"].astype(float)
        ema200_full = ema_no_partial(src, EMA_LEN)
        hma60_full = hma_no_partial(src, HMA_LEN)
        ema200 = ema200_full.reindex(df_view.index)
        hma60 = hma60_full.reindex(df_view.index)

        # Hover text
        ts = df_view.index
        price_hovertext = [
            f"<b>{dt:%d-%b %Y %H:%M}</b><br>"
            f"Open: {float(o):.2f}<br>High: {float(h):.2f}<br>"
            f"Low: {float(l_):.2f}<br>Close: {float(c):.2f}"
            for dt, o, h, l_, c in zip(ts, df_view["Open"], df_view["High"], df_view["Low"], df_view["Close"])
        ]
        vol_hovertext = [
            f"<b>{dt:%d-%b %Y %H:%M}</b><br>Volume: {int(v):,}"
            for dt, v in zip(ts, df_view["Volume"].fillna(0))
        ]

        # Colors
        green_line = "rgba(22,160,133,1)"
        red_line = "rgba(231,76,60,1)"
        green_fill = "rgba(22,160,133,0.8)"
        red_fill = "rgba(231,76,60,0.8)"
        vol_colors = np.where(df_view["Close"] >= df_view["Open"], "rgba(22,160,133,0.45)", "rgba(231,76,60,0.45)")
        ema_color = "#2c7be5"
        hma_color = "#ff7f0e"

        # Figure
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)

        if gapless:
            x_vals = np.arange(len(df_view))
            ticktext = [d.strftime("%d-%b %H:%M") if cfg["intraday"] else d.strftime("%d-%b %Y") for d in ts]

            try:
                fig.add_trace(
                    go.Candlestick(
                        x=x_vals, open=df_view["Open"], high=df_view["High"], low=df_view["Low"], close=df_view["Close"],
                        increasing=dict(line=dict(color=green_line, width=1), fillcolor=green_fill),
                        decreasing=dict(line=dict(color=red_line, width=1), fillcolor=red_fill),
                        whiskerwidth=0,
                        name="Price",
                        hovertemplate="<b>%{text}</b><br>"
                                      "Open: %{open:.2f}<br>High: %{high:.2f}<br>"
                                      "Low: %{low:.2f}<br>Close: %{close:.2f}<extra></extra>",
                        text=ticktext
                    ),
                    row=1, col=1
                )
            except (ValueError, TypeError):
                fig.add_trace(
                    go.Candlestick(
                        x=x_vals, open=df_view["Open"], high=df_view["High"], low=df_view["Low"], close=df_view["Close"],
                        increasing=dict(line=dict(color=green_line, width=1), fillcolor=green_fill),
                        decreasing=dict(line=dict(color=red_line, width=1), fillcolor=red_fill),
                        whiskerwidth=0, name="Price",
                        text=price_hovertext, hoverinfo="text"
                    ),
                    row=1, col=1
                )

            # Overlays
            if not ema200.dropna().empty:
                fig.add_trace(go.Scatter(x=x_vals, y=ema200.values, mode="lines", name=f"EMA {EMA_LEN}",
                                         line=dict(color=ema_color, width=2), hoverinfo="skip"), row=1, col=1)
            if not hma60.dropna().empty:
                fig.add_trace(go.Scatter(x=x_vals, y=hma60.values, mode="lines", name=f"HMA {HMA_LEN}",
                                         line=dict(color=hma_color, width=2), hoverinfo="skip"), row=1, col=1)

            # Volume
            try:
                fig.add_trace(
                    go.Bar(x=x_vals, y=df_view["Volume"], marker=dict(color=vol_colors, line=dict(width=0)),
                           name="Volume",
                           hovertemplate="<b>%{text}</b><br>Volume: %{y:,}<extra></extra>", text=ticktext),
                    row=2, col=1
                )
            except (ValueError, TypeError):
                fig.add_trace(
                    go.Bar(x=x_vals, y=df_view["Volume"], marker=dict(color=vol_colors, line=dict(width=0)),
                           name="Volume", text=vol_hovertext, hoverinfo="text"),
                    row=2, col=1
                )

            for r in (1, 2):
                fig.update_xaxes(type="linear", showgrid=False, ticks="outside",
                                 tickangle=-35, rangeslider=dict(visible=False), row=r, col=1)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(230,230,230,0.45)", zeroline=False,
                                 ticks="outside", row=r, col=1)

            sig_pos_view = sig_pos_full - start_pos_plot
            fig.add_vrect(x0=sig_pos_view-0.5, x1=sig_pos_view+0.5, fillcolor="yellow", opacity=0.14,
                          layer="below", line_width=0, row="all", col=1)

        else:
            intraday = cfg["intraday"]
            range_breaks = []
            if intraday:
                range_breaks = [
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[15.5, 24], pattern="hour"),
                    dict(bounds=[0, 9.25], pattern="hour"),
                ]
            try:
                fig.add_trace(
                    go.Candlestick(
                        x=ts, open=df_view["Open"], high=df_view["High"], low=df_view["Low"], close=df_view["Close"],
                        increasing=dict(line=dict(color=green_line, width=1), fillcolor=green_fill),
                        decreasing=dict(line=dict(color=red_line, width=1), fillcolor=red_fill),
                        whiskerwidth=0,
                        name="Price",
                        hovertemplate="<b>%{x|%d-%b %Y %H:%M}</b><br>"
                                      "Open: %{open:.2f}<br>High: %{high:.2f}<br>"
                                      "Low: %{low:.2f}<br>Close: %{close:.2f}<extra></extra>",
                    ),
                    row=1, col=1
                )
            except (ValueError, TypeError):
                fig.add_trace(
                    go.Candlestick(
                        x=ts, open=df_view["Open"], high=df_view["High"], low=df_view["Low"], close=df_view["Close"],
                        increasing=dict(line=dict(color=green_line, width=1), fillcolor=green_fill),
                        decreasing=dict(line=dict(color=red_line, width=1), fillcolor=red_fill),
                        whiskerwidth=0, name="Price",
                        text=price_hovertext, hoverinfo="text"
                    ),
                    row=1, col=1
                )

            if not ema200.dropna().empty:
                fig.add_trace(go.Scatter(x=ts, y=ema200.values, mode="lines", name=f"EMA {EMA_LEN}",
                                         line=dict(color=ema_color, width=2), hoverinfo="skip"), row=1, col=1)
            if not hma60.dropna().empty:
                fig.add_trace(go.Scatter(x=ts, y=hma60.values, mode="lines", name=f"HMA {HMA_LEN}",
                                         line=dict(color=hma_color, width=2), hoverinfo="skip"), row=1, col=1)

            try:
                fig.add_trace(
                    go.Bar(x=ts, y=df_view["Volume"], marker=dict(color=vol_colors, line=dict(width=0)),
                           name="Volume",
                           hovertemplate="<b>%{x|%d-%b %Y %H:%M}</b><br>Volume: %{y:,}<extra></extra>"),
                    row=2, col=1
                )
            except (ValueError, TypeError):
                fig.add_trace(
                    go.Bar(x=ts, y=df_view["Volume"], marker=dict(color=vol_colors, line=dict(width=0)),
                           name="Volume", text=vol_hovertext, hoverinfo="text"),
                    row=2, col=1
                )

            for r in (1, 2):
                fig.update_xaxes(showgrid=False, tickformat="%d-%b %H:%M" if intraday else "%d-%b %Y",
                                 ticks="outside", tickangle=-35, rangeslider=dict(visible=False),
                                 rangebreaks=range_breaks, row=r, col=1)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(230,230,230,0.45)", zeroline=False,
                                 ticks="outside", row=r, col=1)

            if len(ts) > 1:
                cw_sec = (ts[1] - ts[0]).total_seconds()
            else:
                cw_sec = TIMEFRAME.get(tf)["secs"]
            half = cw_sec / 2.0
            center_dt = candle_start if candle_start in ts else ts[0]
            fig.add_vrect(x0=center_dt - timedelta(seconds=half), x1=center_dt + timedelta(seconds=half),
                          fillcolor="yellow", opacity=0.14, layer="below", line_width=0, row="all", col=1)

        # Layout: axis-drag zoom only (disable wheel & panning)
        fig.update_layout(
            height=int(chart_h),
            margin=dict(l=40, r=20, t=30, b=30),
            template="simple_white",
            hovermode="x unified",
            dragmode="zoom",
            showlegend=True,
            bargap=0.06,
            uirevision=f"{uploaded_name}|{tf}|{gapless}|{auto_fit}|{target_px_per_candle}|{approx_chart_width_px}|{n_before}|{n_after}",
        )
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)

        config = dict(
            displayModeBar=True,
            displaylogo=False,
            scrollZoom=False,  # force zoom via axis-drag (wheel disabled)
            modeBarButtonsToRemove=[
                "toImage","lasso2d","select2d","toggleSpikelines",
                "hoverClosestCartesian","hoverCompareCartesian","pan2d","autoScale2d"
            ],
        )

        sig_title_dt = candle_start if candle_start in df_view.index else df_view.index[0]
        st.subheader(f"{symbol_eq} • TF: {tf} • {sig_title_dt.strftime('%d %b %Y %I:%M %p').lower()} IST")
        st.caption("Drag on the X-axis to zoom horizontally; drag on the Y-axis to zoom vertically. Double-click to reset.")
        st.plotly_chart(fig, use_container_width=True, config=config, key=f"chart_{symbol_eq}_{tf}_{idx}")

        charts_rendered += 1

    # ================== Pagination controls (bottom) ==================
    st.markdown("---")
    shown = min(display_limit, total_rows)
    st.markdown(f"Showing {shown} of {total_rows} charts")

    more_available = display_limit < total_rows
    col1, col2 = st.columns([1, 1])
    with col1:
        if more_available:
            if st.button(f"Load {int(charts_per_click)} more charts"):
                st.session_state.display_limit = min(total_rows, display_limit + int(charts_per_click))
                st.rerun()
        else:
            st.info("All charts from the CSV are displayed.")
    with col2:
        if more_available:
            if st.button("Load all remaining"):
                st.session_state.display_limit = total_rows
                st.rerun()
else:
    st.info("Awaiting screener file…")