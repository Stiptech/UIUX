"""
Stock Price Prediction - Streamlit App
Converted from Jupyter Notebook with model persistence (train once, predict forever)
"""

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── New Figma Design Styling ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --sp-bg: #0B1426;
    --sp-bg-deep: #07111f;
    --sp-card: #0f1b2e;
    --sp-card-2: #1a2942;
    --sp-card-3: #243350;
    --sp-green: #10B981;
    --sp-green-dark: #059669;
    --sp-red: #EF4444;
    --sp-yellow: #F59E0B;
    --sp-text: #E5E7EB;
    --sp-muted: #94A3B8;
    --sp-border: #1F2A3D;
    --sp-radius: 18px;
}

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--sp-bg);
    color: var(--sp-text);
}

[data-testid="stAppViewContainer"] > .main {
    background: var(--sp-bg);
}

.block-container {
    max-width: 1280px;
    padding-top: 1.25rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    background: var(--sp-bg-deep);
    border-right: 1px solid var(--sp-border);
}

section[data-testid="stSidebar"] * {
    color: var(--sp-text);
}

.sp-topnav {
    background: var(--sp-card);
    border: 1px solid var(--sp-border);
    border-radius: 22px;
    padding: 16px 22px;
    margin-bottom: 26px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.sp-brand {
    color: var(--sp-green);
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.03em;
}

.sp-navlinks {
    display: flex;
    gap: 12px;
    align-items: center;
}

.sp-nav-pill {
    background: var(--sp-card-2);
    color: var(--sp-muted);
    border-radius: 999px;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 700;
}

.sp-nav-pill-active {
    background: var(--sp-green);
    color: #ffffff;
}

.sp-icon-row {
    display: flex;
    gap: 10px;
    align-items: center;
}

.sp-icon-btn {
    width: 40px;
    height: 40px;
    border-radius: 999px;
    background: var(--sp-card-2);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--sp-text);
}

.sp-avatar {
    width: 40px;
    height: 40px;
    border-radius: 999px;
    background: linear-gradient(135deg, var(--sp-green), var(--sp-green-dark));
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 800;
}

.sp-section-title {
    font-size: 26px;
    line-height: 1.15;
    font-weight: 800;
    color: white;
    letter-spacing: -0.03em;
    margin: 0 0 18px 0;
}

.sp-card {
    background: var(--sp-card);
    border: 1px solid var(--sp-border);
    border-radius: var(--sp-radius);
    padding: 24px;
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.18);
}

.sp-card-soft {
    background: var(--sp-card-2);
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    padding: 16px;
}

.sp-stock-name {
    color: var(--sp-muted);
    font-size: 14px;
    margin-bottom: 4px;
}

.sp-price {
    color: white;
    font-size: 46px;
    line-height: 1;
    font-weight: 900;
    letter-spacing: -0.05em;
}

.sp-change-up {
    color: var(--sp-green);
    font-size: 14px;
    font-weight: 700;
    margin-top: 8px;
}

.sp-change-down {
    color: var(--sp-red);
    font-size: 14px;
    font-weight: 700;
    margin-top: 8px;
}

.sp-muted {
    color: var(--sp-muted);
}

.sp-chip-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 18px 0;
}

.sp-chip {
    background: var(--sp-card-2);
    color: var(--sp-muted);
    border-radius: 10px;
    padding: 9px 15px;
    font-size: 13px;
    font-weight: 700;
}

.sp-chip-active {
    background: var(--sp-green);
    color: white;
}

.sp-small-label {
    color: var(--sp-muted);
    font-size: 12px;
    margin-bottom: 5px;
}

.sp-stat-value {
    color: white;
    font-size: 20px;
    font-weight: 800;
}

.sp-rec-title {
    color: var(--sp-green);
    font-size: 28px;
    font-weight: 900;
    line-height: 1;
}

.sp-badge {
    background: var(--sp-card-2);
    color: var(--sp-muted);
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 800;
}

.sp-badge-green {
    background: rgba(16, 185, 129, 0.16);
    color: var(--sp-green);
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 800;
}

.sp-insight {
    background: var(--sp-card-2);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 12px;
}

.sp-insight-label {
    color: var(--sp-green);
    font-size: 12px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sp-insight-text {
    color: #CBD5E1;
    font-size: 14px;
    line-height: 1.55;
}

.sp-watch-row {
    background: var(--sp-card-2);
    border-radius: 16px;
    padding: 15px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.sp-symbol-circle {
    width: 44px;
    height: 44px;
    border-radius: 999px;
    background: linear-gradient(135deg, var(--sp-green), var(--sp-green-dark));
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    color: white;
}

.sp-flex {
    display: flex;
    align-items: center;
    gap: 12px;
}

.sp-disclaimer {
    background: rgba(245, 158, 11, 0.09);
    border: 1px solid rgba(245, 158, 11, 0.25);
    color: #FCD34D;
    border-radius: 14px;
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.5;
}

.stButton > button {
    background: var(--sp-green) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.8rem 1rem !important;
    font-weight: 800 !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: var(--sp-green-dark) !important;
    color: white !important;
    border: none !important;
}

.stTextInput input, .stSelectbox div[data-baseweb="select"], .stSlider {
    color: var(--sp-text) !important;
}

.stTextInput input {
    background: var(--sp-card-2) !important;
    border: 1px solid var(--sp-border) !important;
    border-radius: 12px !important;
}

[data-testid="stMetric"] {
    background: var(--sp-card-2);
    border: 1px solid var(--sp-border);
    border-radius: 14px;
    padding: 14px;
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

hr {
    border-color: var(--sp-border);
}

@media (max-width: 768px) {
    .sp-topnav {
        align-items: flex-start;
        gap: 14px;
        flex-direction: column;
    }
    .sp-navlinks {
        overflow-x: auto;
        width: 100%;
        padding-bottom: 4px;
    }
    .sp-price {
        font-size: 38px;
    }
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
PAST_DAYS_TARGET = 5   # days ahead to predict target
PAST_DAYS_LAG    = 20  # lookback window
N_TARGETS        = 5   # number of output days
MODEL_DIR        = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Data & Processing Functions ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker: str) -> pd.DataFrame:
    """Download stock data for the last 2 years."""
    tahun = datetime.now().year
    tanggal_awal = f"{tahun - 1}-01-01"
    tanggal_akhir = f"{tahun}-12-31"
    df = yf.download(ticker, start=tanggal_awal, end=tanggal_akhir, progress=False)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Please check the symbol.")
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[["Date", "Close", "High", "Low", "Open", "Volume"]].copy()
    df["Date"] = df["Date"].dt.strftime("%Y%m%d").astype("int64")
    df["Volume"] = df["Volume"].astype("float64")
    return df


def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply outlier treatment from the notebook."""
    df = df.copy()
    filt = df.index.isin(df[(np.abs(stats.zscore(df["High"])) > 3)].index.values)
    df.loc[filt, ["Open", "High", "Low", "Close"]] = df.loc[
        filt, ["Open", "High", "Low", "Close"]
    ].apply(lambda x: x / 100)
    return df


def time_lag_transform(data: pd.DataFrame, past_days: int, suffix: str = "+") -> pd.DataFrame:
    """Create time-lag features (forward or backward shift)."""
    data = data.copy()
    cols = data.columns.tolist()
    new_cols = {}
    for i in range(past_days):
        for j in cols:
            if suffix == "+":
                new_cols[f"{j}+{i + 1}"] = data[j].shift(periods=-(i))
            else:
                new_cols[f"{j}-{i}"] = data[j].shift(periods=i)
    result = pd.concat([data, pd.DataFrame(new_cols, index=data.index)], axis=1)
    if suffix == "+":
        return result.drop(columns=cols, axis=1)
    else:
        return result.drop(columns=cols, axis=1)


def prepare_features(df: pd.DataFrame):
    """
    Full preprocessing pipeline (mirrors the notebook exactly).
    Returns X (features), y (targets), scaler_x, scaler_y, df_ori.
    """
    from sklearn.preprocessing import StandardScaler

    df = treat_outliers(df)
    df_ori = df.copy().drop("Date", axis=1)

    # ── Target Generator ──────────────────────────────────────────────────────
    df["Target"] = (df["High"] + df["Low"]) / 2

    # ── Target time-lag (next 5 days) ────────────────────────────────────────
    df_temp = df[["Target"]].copy()
    future_cols = {}
    for i in range(PAST_DAYS_TARGET):
        future_cols[f"Target+{i + 1}"] = df_temp["Target"].shift(periods=-(i + 1))
    df_temp = pd.DataFrame(future_cols, index=df_temp.index).dropna()
    target_col = df_temp.columns.tolist()

    # Join targets
    delta_row = len(df) - len(df_temp)
    df = pd.concat([df, df_temp], axis=1, join="inner").drop("Target", axis=1)

    # ── Drop date column ───────────────────────────────────────────────────────
    df.drop("Date", axis=1, inplace=True)

    # ── Lookback time-lag (past 20 days) ─────────────────────────────────────
    feature_base = df.drop(columns=target_col)
    new_lag_cols = {}
    base_cols = feature_base.columns.tolist()
    for i in reversed(range(PAST_DAYS_LAG)):
        for j in base_cols:
            new_lag_cols[f"{j}-{i}"] = feature_base[j].shift(periods=i)
    df_lag = pd.DataFrame(new_lag_cols, index=feature_base.index).dropna()

    # Join with targets
    df = pd.concat([df_lag, df[target_col]], axis=1, join="inner")

    # ── Train / Val split (50/50) ─────────────────────────────────────────────
    panjang = len(df)
    valid_point = round(0.5 * panjang)

    # Split X / y
    X = df.iloc[:, :-N_TARGETS].values
    y = df.iloc[:, -N_TARGETS:].values

    X_train, X_valid = X[:valid_point], X[valid_point:]
    y_train, y_valid = y[:valid_point], y[valid_point:]

    # ── Normalize ─────────────────────────────────────────────────────────────
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_x.fit_transform(X_train)
    X_valid_s = scaler_x.transform(X_valid)
    y_train_s = scaler_y.fit_transform(y_train)
    y_valid_s = scaler_y.transform(y_valid)

    # ── Reshape for Conv1D: (samples, timesteps, features) ──────────────────
    n_features = int(X_train_s.shape[1] / PAST_DAYS_LAG)
    X_train_s = X_train_s.reshape(X_train_s.shape[0], PAST_DAYS_LAG, n_features)
    X_valid_s = X_valid_s.reshape(X_valid_s.shape[0], PAST_DAYS_LAG, n_features)

    return (
        X_train_s, y_train_s,
        X_valid_s, y_valid_s,
        scaler_x, scaler_y,
        df_ori, df.columns[-N_TARGETS:].tolist()
    )


def build_model(input_shape, n_outputs):
    """Build the Conv1D model (same architecture as notebook)."""
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import MeanAbsolutePercentageError

    inputs  = tf.keras.layers.Input(shape=input_shape)
    cnn     = tf.keras.layers.Conv1D(filters=200, kernel_size=PAST_DAYS_LAG)(inputs)
    do      = tf.keras.layers.Dropout(0.1)(cnn)
    mlp     = tf.keras.layers.Dense(200)(do)
    flatten = tf.keras.layers.Flatten()(mlp)
    outputs = tf.keras.layers.Dense(n_outputs)(flatten)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(
        loss="mean_squared_error",
        optimizer=Adam(learning_rate=1e-3),
        metrics=[MeanAbsolutePercentageError(name="mape")],
    )
    return model


def model_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker.replace('.', '_')}.keras")


def scaler_path(ticker: str) -> str:
    return os.path.join(MODEL_DIR, f"{ticker.replace('.', '_')}_scalers.pkl")


def model_exists(ticker: str) -> bool:
    return os.path.exists(model_path(ticker)) and os.path.exists(scaler_path(ticker))



def save_model_and_scalers(model, scaler_x, scaler_y, ticker: str):
    model.save(model_path(ticker))
    with open(scaler_path(ticker), "wb") as f:
        pickle.dump({"scaler_x": scaler_x, "scaler_y": scaler_y}, f)


def load_model_and_scalers(ticker: str):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path(ticker))
    with open(scaler_path(ticker), "rb") as f:
        scalers = pickle.load(f)
    return model, scalers["scaler_x"], scalers["scaler_y"]


def get_latest_input(ticker: str, scaler_x):
    """
    Fetch the most recent data and shape it into a single prediction input.
    """
    tahun = datetime.now().year
    df = yf.download(ticker, start=f"{tahun - 1}-01-01", end=f"{tahun}-12-31", progress=False)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df[["Date", "Close", "High", "Low", "Open", "Volume"]].copy()
    df["Date"] = df["Date"].dt.strftime("%Y%m%d").astype("int64")
    df["Volume"] = df["Volume"].astype("float64")
    df = treat_outliers(df)
    df.drop("Date", axis=1, inplace=True)

    # We need the last PAST_DAYS_LAG rows for one prediction
    if len(df) < PAST_DAYS_LAG:
        raise ValueError("Not enough recent data for prediction.")

    recent = df.tail(PAST_DAYS_LAG).values  # shape: (20, 5)
    # Build one lag-feature row: flatten in reverse (oldest first)
    # The notebook builds lag features as: col-19, col-18, ..., col-0
    # = row[0], row[1], ..., row[19] for each column
    feature_row = []
    cols = ["Close", "High", "Low", "Open", "Volume"]
    # recent sudah berurutan dari data paling lama ke data terbaru.
    # Urutan ini harus sama dengan fitur training: hari lama menuju hari terbaru.
    for row in recent:
        for c_idx in range(len(cols)):
            feature_row.append(row[c_idx])

    X_raw = np.array(feature_row).reshape(1, -1)
    X_scaled = scaler_x.transform(X_raw)
    n_features = int(X_scaled.shape[1] / PAST_DAYS_LAG)
    X_scaled = X_scaled.reshape(1, PAST_DAYS_LAG, n_features)
    return X_scaled, recent


def predict_next_days(model, scaler_y, X_input):
    """Run model and inverse-transform."""
    y_pred_scaled = model.predict(X_input, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return y_pred[0]



# ─── UI ───────────────────────────────────────────────────────────────────────
def _fmt_price(value, ticker):
    try:
        value = float(value)
    except Exception:
        return str(value)
    if ticker.endswith(".JK"):
        return f"Rp {value:,.2f}"
    return f"${value:,.2f}"


def _fmt_volume(value):
    try:
        value = float(value)
    except Exception:
        return str(value)
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:,.0f}"


def _signal_from_change(change_pct):
    if change_pct >= 1:
        return "BUY", "Strong upward movement detected from latest market data. Use this as learning output, not investment advice."
    if change_pct <= -1:
        return "SELL", "Downward pressure detected from latest market data. Review risk before acting."
    return "HOLD", "Movement is still narrow. Wait for clearer direction before making a decision."


def _top_nav():
    st.markdown("""
    <div class="sp-topnav">
        <div class="sp-flex">
            <div class="sp-brand">StockPredict AI</div>
            <div class="sp-navlinks">
                <div class="sp-nav-pill sp-nav-pill-active">Home</div>
                <div class="sp-nav-pill">Watchlist</div>
                <div class="sp-nav-pill">Alerts</div>
                <div class="sp-nav-pill">AI Insights</div>
            </div>
        </div>
        <div class="sp-icon-row">
            <div class="sp-icon-btn">⌕</div>
            <div class="sp-icon-btn">🔔</div>
            <div class="sp-avatar">AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _stock_header(df_raw, ticker):
    last_close = df_raw["Close"].iloc[-1]
    prev_close = df_raw["Close"].iloc[-2]
    change = last_close - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
    change_class = "sp-change-up" if change >= 0 else "sp-change-down"
    arrow = "↗" if change >= 0 else "↘"
    high_period = df_raw["High"].max()
    low_period = df_raw["Low"].min()
    avg_volume = df_raw["Volume"].mean()

    st.markdown(f"""
    <div class="sp-card">
        <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:16px;">
            <div>
                <div class="sp-stock-name">Selected Stock</div>
                <div class="sp-price">{_fmt_price(last_close, ticker)}</div>
                <div class="{change_class}">{arrow} {_fmt_price(abs(change), ticker)} ({change_pct:+.2f}%) today</div>
            </div>
            <div style="text-align:right;">
                <div class="sp-small-label">Ticker</div>
                <div style="font-weight:800; font-size:18px; color:white;">{ticker}</div>
                <div class="sp-muted" style="font-size:12px; margin-top:5px;">Last data loaded</div>
            </div>
        </div>
        <div class="sp-chip-row">
            <div class="sp-chip">1D</div>
            <div class="sp-chip">1W</div>
            <div class="sp-chip sp-chip-active">1M</div>
            <div class="sp-chip">3M</div>
            <div class="sp-chip">1Y</div>
            <div class="sp-chip">All</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    chart_df = df_raw.copy()
    chart_df["Date_str"] = chart_df["Date"].astype(str)
    chart_df["Date_dt"] = pd.to_datetime(chart_df["Date_str"], format="%Y%m%d")
    chart_display = chart_df.set_index("Date_dt")[["Close"]].rename(columns={"Close": "Close Price"})
    st.line_chart(chart_display, color=["#10B981"], height=310)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"<div class='sp-card-soft'><div class='sp-small-label'>Low</div><div class='sp-stat-value'>{_fmt_price(low_period, ticker)}</div></div>", unsafe_allow_html=True)
    with s2:
        st.markdown(f"<div class='sp-card-soft'><div class='sp-small-label'>High</div><div class='sp-stat-value'>{_fmt_price(high_period, ticker)}</div></div>", unsafe_allow_html=True)
    with s3:
        st.markdown(f"<div class='sp-card-soft'><div class='sp-small-label'>Data Points</div><div class='sp-stat-value'>{len(df_raw)}</div></div>", unsafe_allow_html=True)
    with s4:
        st.markdown(f"<div class='sp-card-soft'><div class='sp-small-label'>Avg Volume</div><div class='sp-stat-value'>{_fmt_volume(avg_volume)}</div></div>", unsafe_allow_html=True)

    return last_close, prev_close, change, change_pct


def _right_panel(ticker, change_pct):
    signal, reason = _signal_from_change(change_pct)
    badge_class = "sp-badge-green" if signal == "BUY" else "sp-badge"
    st.markdown(f"""
    <div class="sp-card">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
            <div class="sp-rec-title">{signal}</div>
            <div class="{badge_class}">AI Signal</div>
        </div>
        <div style="font-weight:800; margin-bottom:10px;">Why this pick?</div>
        <div class="sp-insight-text" style="margin-bottom:18px;">{reason}</div>
        <div class="sp-disclaimer">Prediction output is for academic purposes. Do not present this as financial advice.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sp-card" style="margin-top:18px;">
        <h3 style="margin-top:0; margin-bottom:16px;">Market Intelligence</h3>
        <div class="sp-insight">
            <div class="sp-insight-label">Good to know</div>
            <div class="sp-insight-text">The chart uses Yahoo Finance historical data for {ticker}.</div>
        </div>
        <div class="sp-insight">
            <div class="sp-insight-label" style="color:#F59E0B;">Model note</div>
            <div class="sp-insight-text">Train the model once. Saved models make later prediction faster.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _watchlist_section():
    stocks = [
        ("BBCA.JK", "Bank Central Asia", "+0.85%", True),
        ("TLKM.JK", "Telkom Indonesia", "+0.42%", True),
        ("GOTO.JK", "GoTo Gojek Tokopedia", "-0.21%", False),
        ("BMRI.JK", "Bank Mandiri", "+0.58%", True),
    ]
    st.markdown('<div class="sp-section-title" style="margin-top:26px;">Watchlist</div>', unsafe_allow_html=True)
    for symbol, name, change, positive in stocks:
        cls = "sp-change-up" if positive else "sp-change-down"
        st.markdown(f"""
        <div class="sp-watch-row">
            <div class="sp-flex">
                <div class="sp-symbol-circle">{symbol[0]}</div>
                <div>
                    <div style="font-weight:800; color:white;">{symbol}</div>
                    <div class="sp-muted" style="font-size:13px;">{name}</div>
                </div>
            </div>
            <div class="{cls}" style="margin-top:0;">{change}</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    _top_nav()

    with st.sidebar:
        st.markdown("### Configuration")
        ticker = st.text_input(
            "Stock Ticker Symbol",
            value="GOTO.JK",
            placeholder="Example: GOTO.JK, BBCA.JK, AAPL",
            help="Indonesian stocks use .JK suffix. Example: BBCA.JK"
        ).upper().strip()

        epochs_train = st.slider("Training Epochs", 20, 200, 100, 10)
        epochs_finetune = st.slider("Fine-tune Epochs", 5, 50, 10, 5)

        st.markdown("---")
        btn_predict = False
        btn_retrain = False
        btn_train = False

        if ticker:
            if model_exists(ticker):
                st.success(f"Model ready for {ticker}")
                btn_predict = st.button("Predict Now", key="predict_saved")
                btn_retrain = st.button("Retrain", key="retrain")
            else:
                st.warning(f"No saved model for {ticker}")
                btn_train = st.button("Train and Predict", key="train_new")

        st.markdown("---")
        st.caption("This dashboard adapts the new Figma mobile design into Streamlit layout.")

    if not ticker:
        st.info("Enter a stock ticker in the sidebar.")
        return

    df_raw = None
    try:
        with st.spinner(f"Loading market data for {ticker}..."):
            df_raw = download_stock_data(ticker)
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown('<div class="sp-section-title">Stock Detail</div>', unsafe_allow_html=True)
        last_close, prev_close, change, change_pct = _stock_header(df_raw, ticker)
    with right:
        _right_panel(ticker, change_pct)

    _watchlist_section()

    action = None
    if model_exists(ticker):
        if btn_predict:
            action = "predict"
        elif btn_retrain:
            action = "retrain"
    else:
        if btn_train:
            action = "train"

    if action == "predict":
        _run_prediction(ticker)
    elif action in ("train", "retrain"):
        _run_training(ticker, df_raw, epochs_train, epochs_finetune)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        if model_exists(ticker):
            st.info("Saved model found. Use Predict Now in the sidebar to show the 5-day forecast.")
        else:
            st.info("No saved model yet. Use Train and Predict in the sidebar to create the first model.")

    with st.expander("View Raw Data"):
        display_df = df_raw.tail(20).copy()
        display_df["Date"] = pd.to_datetime(display_df["Date"].astype(str), format="%Y%m%d")
        st.dataframe(
            display_df.set_index("Date").style.format(
                {"Close": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}", "Open": "{:.2f}", "Volume": "{:,.0f}"}
            ),
            use_container_width=True,
        )


def _run_training(ticker, df_raw, epochs_train, epochs_finetune):
    """Train, save, and then predict."""
    import tensorflow as tf

    st.markdown('<div class="section-header"><h3>🏋️ Model Training</h3></div>', unsafe_allow_html=True)

    progress_bar = st.progress(0, text="Preparing data…")
    status_box   = st.empty()

    try:
        # ── 1. Prepare data ──────────────────────────────────────────────────
        status_box.info("⚙️ Preprocessing data and creating features…")
        (X_train, y_train,
         X_valid, y_valid,
         scaler_x, scaler_y,
         df_ori, target_col) = prepare_features(df_raw)

        progress_bar.progress(15, text="Data prepared ✓")
        status_box.info(
            f"📐 Shapes → X_train: {X_train.shape}, y_train: {y_train.shape} | "
            f"X_valid: {X_valid.shape}, y_valid: {y_valid.shape}"
        )

        # ── 2. Build model ───────────────────────────────────────────────────
        model = build_model(X_train.shape[1:], N_TARGETS)
        progress_bar.progress(20, text="Model built ✓")

        # ── 3. Train (pass 1: on training data) ─────────────────────────────
        status_box.info(f"🚀 Training for {epochs_train} epochs on training set…")

        history_train = model.fit(
            X_train, y_train,
            epochs=epochs_train,
            batch_size=1,
            validation_data=(X_valid, y_valid),
            verbose=0,
        )
        progress_bar.progress(65, text="Pass 1 complete ✓")

        # ── 4. Fine-tune (pass 2: on validation data) ───────────────────────
        status_box.info(f"🔧 Fine-tuning for {epochs_finetune} epochs on validation set…")
        model.fit(
            X_valid, y_valid,
            epochs=epochs_finetune,
            batch_size=1,
            verbose=0,
        )
        progress_bar.progress(85, text="Fine-tune complete ✓")

        # ── 5. Save model + scalers ──────────────────────────────────────────
        save_model_and_scalers(model, scaler_x, scaler_y, ticker)
        progress_bar.progress(95, text="Model saved ✓")

        status_box.success(f"✅ Model trained and saved to `{model_path(ticker)}`")

        # ── 6. Show training metrics ─────────────────────────────────────────
        final_val_loss = history_train.history["val_loss"][-1]
        final_val_mape = history_train.history.get("val_mape", [None])[-1]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Val Loss (MSE)", f"{final_val_loss:.4f}")
        with col2:
            if final_val_mape:
                st.metric("Final Val MAPE", f"{final_val_mape:.2f}%")

        # Loss curve
        loss_df = pd.DataFrame({
            "Training Loss": history_train.history["loss"],
            "Validation Loss": history_train.history["val_loss"],
        })
        st.markdown("**Training Loss Curve**")
        st.line_chart(loss_df, color=["#58a6ff", "#f85149"], height=220)

        progress_bar.progress(100, text="Done!")

    except Exception as e:
        status_box.error(f"❌ Training failed: {e}")
        st.exception(e)
        return

    # ── 7. Predict ───────────────────────────────────────────────────────────
    _run_prediction(ticker)


def _run_prediction(ticker):
    """Load saved model and predict next 5 days."""
    st.markdown('<div class="section-header"><h3>🔮 5-Day Price Forecast</h3></div>', unsafe_allow_html=True)

    with st.spinner("Loading model and computing predictions…"):
        try:
            model, scaler_x, scaler_y = load_model_and_scalers(ticker)
            X_latest, recent_data = get_latest_input(ticker, scaler_x)
            predictions = predict_next_days(model, scaler_y, X_latest)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)
            return

    # Reference: last known mid-price
    last_close  = recent_data[-1][0]   # Close of latest row
    last_mid    = (recent_data[-1][1] + recent_data[-1][2]) / 2   # (High+Low)/2

    st.markdown(f"""
    <div class="info-box">
        📌 Reference: Last Close = <strong>{last_close:.2f}</strong> &nbsp;|&nbsp;
        Last Mid-Price (High+Low)/2 = <strong>{last_mid:.2f}</strong><br>
        Predictions represent the estimated mid-price for each of the next 5 trading days.
    </div>
    """, unsafe_allow_html=True)

    # Prediction cards
    cols = st.columns(5)
    for i, (col, pred) in enumerate(zip(cols, predictions)):
        chg     = pred - last_mid
        chg_pct = (chg / last_mid) * 100 if last_mid != 0 else 0
        arrow   = "▲" if chg >= 0 else "▼"
        css_cls = "pred-up" if chg >= 0 else "pred-down"
        with col:
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-day">Day +{i + 1}</div>
                <div class="pred-price">{pred:.2f}</div>
                <div class="pred-change {css_cls}">{arrow} {abs(chg_pct):.2f}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast chart
    days = [f"Day +{i + 1}" for i in range(5)]
    pred_df = pd.DataFrame({"Predicted Mid-Price": predictions}, index=days)
    st.markdown("**Forecast Trend**")
    st.line_chart(pred_df, color=["#79c0ff"], height=220)

    # Summary table
    st.markdown("**Detailed Forecast Table**")
    summary = pd.DataFrame({
        "Day":              days,
        "Predicted Price":  [f"{p:.4f}" for p in predictions],
        "Change from Mid":  [f"{(p - last_mid):+.4f}" for p in predictions],
        "Change %":         [f"{((p - last_mid) / last_mid * 100):+.2f}%" for p in predictions],
        "Signal":           ["🟢 BUY" if p > last_mid else "🔴 SELL" for p in predictions],
    })
    st.dataframe(summary.set_index("Day"), use_container_width=True)

    st.markdown("""
    <div class="info-box" style="font-size:0.78rem; color:#8b949e;">
        ⚠️ <strong>Disclaimer:</strong> Predictions are generated by an AI model and are for
        educational purposes only. Past performance does not guarantee future results.
        Always consult a financial professional before making investment decisions.
    </div>
    """, unsafe_allow_html=True)




if __name__ == "__main__":
    main()
