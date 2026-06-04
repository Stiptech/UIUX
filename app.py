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
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium Design Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Space+Mono:wght@400;700&display=swap');

:root {
    --sp-bg: #080E1A;
    --sp-bg-deep: #050A12;
    --sp-card: #0D1625;
    --sp-card-2: #131E30;
    --sp-card-3: #1B2A40;
    --sp-card-hover: #172035;
    --sp-green: #00D084;
    --sp-green-dark: #00A368;
    --sp-green-glow: rgba(0, 208, 132, 0.15);
    --sp-red: #FF4D6A;
    --sp-yellow: #FFB547;
    --sp-blue: #3B82F6;
    --sp-text: #E2E8F0;
    --sp-muted: #64748B;
    --sp-muted-light: #94A3B8;
    --sp-border: rgba(255,255,255,0.06);
    --sp-border-strong: rgba(255,255,255,0.10);
    --sp-radius: 20px;
    --sp-radius-sm: 14px;
    --font-display: 'DM Sans', sans-serif;
    --font-mono: 'Space Mono', monospace;
}

*, *::before, *::after {
    font-family: var(--font-display);
    box-sizing: border-box;
}

.stApp {
    background: var(--sp-bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,208,132,0.06) 0%, transparent 70%);
    color: var(--sp-text);
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

.block-container {
    max-width: 1320px;
    padding-top: 1.5rem;
    padding-bottom: 5rem;
    padding-left: 2rem;
    padding-right: 2rem;
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

section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--sp-muted-light);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--sp-border);
}

/* ── Brand text kept for fallback references ──────────── */
.sp-brand {
    color: var(--sp-green);
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* ── Section Title ────────────────────────────────────── */
.sp-section-title {
    font-size: 24px;
    line-height: 1.2;
    font-weight: 700;
    color: white;
    letter-spacing: -0.025em;
    margin: 0 0 20px 0;
}

.sp-section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--sp-green);
    margin-bottom: 8px;
}

/* ── Cards ────────────────────────────────────────────── */
.sp-card {
    background: var(--sp-card);
    border: 1px solid var(--sp-border-strong);
    border-radius: var(--sp-radius);
    padding: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.2s;
}

.sp-card:hover {
    border-color: rgba(255,255,255,0.12);
}

.sp-card-soft {
    background: var(--sp-card-2);
    border: 1px solid var(--sp-border);
    border-radius: var(--sp-radius-sm);
    padding: 16px 18px;
    transition: background 0.2s;
}

.sp-card-soft:hover {
    background: var(--sp-card-3);
}

/* ── Price Display ────────────────────────────────────── */
.sp-stock-name {
    color: var(--sp-muted);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.sp-price {
    color: white;
    font-size: 44px;
    line-height: 1;
    font-weight: 700;
    letter-spacing: -0.04em;
    font-family: var(--font-mono);
}

.sp-change-up {
    color: var(--sp-green);
    font-size: 14px;
    font-weight: 600;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 4px;
}

.sp-change-down {
    color: var(--sp-red);
    font-size: 14px;
    font-weight: 600;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 4px;
}

.sp-muted {
    color: var(--sp-muted);
}

/* ── Chips ────────────────────────────────────────────── */
.sp-chip-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin: 16px 0;
}

.sp-chip {
    background: var(--sp-card-2);
    color: var(--sp-muted-light);
    border: 1px solid var(--sp-border);
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
}

.sp-chip:hover {
    background: var(--sp-card-3);
    color: white;
}

.sp-chip-active {
    background: var(--sp-green-glow);
    color: var(--sp-green);
    border-color: rgba(0,208,132,0.3);
}

/* ── Stats ────────────────────────────────────────────── */
.sp-small-label {
    color: var(--sp-muted);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.sp-stat-value {
    color: white;
    font-size: 18px;
    font-weight: 700;
    font-family: var(--font-mono);
    letter-spacing: -0.02em;
}

/* ── Signal / Recommendation ──────────────────────────── */
.sp-rec-title {
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.04em;
}

.sp-rec-buy { color: var(--sp-green); }
.sp-rec-sell { color: var(--sp-red); }
.sp-rec-hold { color: var(--sp-yellow); }

.sp-badge {
    background: var(--sp-card-3);
    color: var(--sp-muted-light);
    border: 1px solid var(--sp-border);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.sp-badge-green {
    background: var(--sp-green-glow);
    color: var(--sp-green);
    border: 1px solid rgba(0,208,132,0.25);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Insights ─────────────────────────────────────────── */
.sp-insight {
    background: var(--sp-card-2);
    border: 1px solid var(--sp-border);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}

.sp-insight:hover {
    border-color: rgba(0,208,132,0.2);
}

.sp-insight-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.sp-insight-label-green { color: var(--sp-green); }
.sp-insight-label-amber { color: var(--sp-yellow); }

.sp-insight-text {
    color: var(--sp-muted-light);
    font-size: 13.5px;
    line-height: 1.6;
}

/* ── Watchlist ─────────────────────────────────────────── */
.sp-watch-row {
    background: var(--sp-card-2);
    border: 1px solid var(--sp-border);
    border-radius: 16px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    transition: all 0.2s;
}

.sp-watch-row:hover {
    background: var(--sp-card-3);
    border-color: var(--sp-border-strong);
}

.sp-symbol-circle {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    background: linear-gradient(135deg, rgba(0,208,132,0.2), rgba(0,163,104,0.1));
    border: 1px solid rgba(0,208,132,0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: var(--sp-green);
    font-size: 15px;
}

.sp-flex {
    display: flex;
    align-items: center;
    gap: 12px;
}

/* ── Section & Info ────────────────────────────────────── */
.section-header {
    margin: 28px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-header h3 {
    color: white;
    font-size: 18px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}

.section-header-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--sp-green);
    flex-shrink: 0;
}

.info-box {
    background: var(--sp-card-2);
    border: 1px solid var(--sp-border-strong);
    border-left: 3px solid var(--sp-green);
    border-radius: 14px;
    padding: 14px 18px;
    font-size: 13.5px;
    color: var(--sp-muted-light);
    line-height: 1.65;
    margin-bottom: 20px;
}

/* ── Prediction Cards ──────────────────────────────────── */
.pred-card {
    background: var(--sp-card-2);
    border: 1px solid var(--sp-border);
    border-radius: 16px;
    padding: 20px 14px;
    text-align: center;
    margin-bottom: 12px;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
}

.pred-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--sp-green), transparent);
    opacity: 0;
    transition: opacity 0.2s;
}

.pred-card:hover {
    border-color: rgba(0,208,132,0.2);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

.pred-card:hover::before {
    opacity: 1;
}

.pred-day {
    color: var(--sp-muted);
    font-size: 10px;
    font-weight: 600;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.pred-price {
    color: white;
    font-size: 20px;
    font-weight: 700;
    font-family: var(--font-mono);
    letter-spacing: -0.02em;
    margin-bottom: 8px;
    line-height: 1.2;
}

.pred-change {
    font-size: 12px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-block;
}

.pred-up {
    color: var(--sp-green);
    background: rgba(0,208,132,0.1);
}

.pred-down {
    color: var(--sp-red);
    background: rgba(255,77,106,0.1);
}

/* ── Notification Panel ────────────────────────────────── */
.sp-notif-panel {
    background: var(--sp-card);
    border: 1px solid var(--sp-border-strong);
    border-radius: 18px;
    padding: 16px 20px;
    margin-bottom: 16px;
    max-width: 380px;
    box-shadow: 0 16px 48px rgba(0,0,0,0.4);
}

.sp-notif-item {
    color: var(--sp-text);
    font-size: 13px;
    padding: 8px 0;
    border-bottom: 1px solid var(--sp-border);
    display: flex;
    align-items: center;
    gap: 8px;
}

.sp-notif-item:last-child {
    border-bottom: none;
}

/* ── Disclaimer ────────────────────────────────────────── */
.sp-disclaimer {
    background: rgba(255,181,71,0.06);
    border: 1px solid rgba(255,181,71,0.2);
    color: #F5D58A;
    border-radius: 14px;
    padding: 14px 16px;
    font-size: 13px;
    line-height: 1.55;
}

/* ── Streamlit Overrides ──────────────────────────────── */
.stButton > button {
    background: var(--sp-green) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1rem !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    width: 100% !important;
    font-family: var(--font-display) !important;
    letter-spacing: -0.01em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(0,208,132,0.25) !important;
}

.stButton > button:hover {
    background: var(--sp-green-dark) !important;
    box-shadow: 0 6px 20px rgba(0,208,132,0.35) !important;
    transform: translateY(-1px) !important;
}

.stTextInput input, .stSelectbox div[data-baseweb="select"] {
    color: var(--sp-text) !important;
}

.stTextInput input {
    background: var(--sp-card-2) !important;
    border: 1px solid var(--sp-border-strong) !important;
    border-radius: 12px !important;
    font-family: var(--font-display) !important;
}

.stTextInput input:focus {
    border-color: rgba(0,208,132,0.4) !important;
    box-shadow: 0 0 0 3px rgba(0,208,132,0.1) !important;
}

[data-testid="stMetric"] {
    background: var(--sp-card-2);
    border: 1px solid var(--sp-border);
    border-radius: 16px;
    padding: 16px 18px;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--sp-border) !important;
}

hr {
    border-color: var(--sp-border);
    margin: 20px 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--sp-bg-deep); }
::-webkit-scrollbar-thumb { background: var(--sp-card-3); border-radius: 99px; }

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
        font-size: 34px;
    }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
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
    tanggal_awal = f"{tahun - 2}-01-01"
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
    df = yf.download(ticker, start=f"{tahun - 2}-01-01", end=f"{tahun}-12-31", progress=False)
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
    """Single-bar navbar built entirely with styled Streamlit buttons."""
    if "nav_tab"    not in st.session_state: st.session_state.nav_tab    = "Home"
    if "notif_open" not in st.session_state: st.session_state.notif_open = False
    if "logged_in"  not in st.session_state: st.session_state.logged_in  = False
    if "login_user" not in st.session_state: st.session_state.login_user = ""

    NAV_TABS     = ["Home", "Watchlist", "Alerts", "AI Insights"]
    active       = st.session_state.nav_tab
    notif_open   = st.session_state.notif_open
    avatar_label = st.session_state.login_user[:2].upper() if st.session_state.logged_in else "AI"

    # ── Per-button dynamic CSS ────────────────────────────────────────────────
    # Build CSS for each nav button: active tab gets green pill, others transparent.
    tab_styles = ""
    for tab in NAV_TABS:
        key = f"nav_{tab}"
        if tab == active:
            tab_styles += f"""
            div[data-testid="stButton"] > button[data-testid="baseButton-secondary"][kind="secondary"]#btn_{key},
            div:has(> button[key="{key}"]) > button {{}}
            """
        # Use a data attribute trick via unique aria-label selectors
    # We use nth-child targeting based on column position instead:
    # col layout: [brand(HTML) | Home | Watchlist | Alerts | AI Insights | 🔔 | 👤]
    # Simpler: inject a class on the container and target button by order.

    # Build per-key CSS using button label text selectors:
    active_bg   = "#00D084"
    inactive_bg = "transparent"
    btn_css = f"""
    <style>
    /* ── Navbar container ─────────────────────────────── */
    .navbar-row {{
        background: #0D1625;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 8px 16px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
    }}
    /* ── Kill default button chrome for ALL nav buttons ─ */
    .navbar-row .stButton > button {{
        border: none !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        padding: 7px 18px !important;
        width: 100% !important;
        background: {inactive_bg} !important;
        color: #64748B !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }}
    .navbar-row .stButton > button:hover {{
        background: rgba(255,255,255,0.05) !important;
        color: #E2E8F0 !important;
        box-shadow: none !important;
        transform: none !important;
    }}
    /* ── Active tab: green pill ──────────────────────── */
    .navbar-row .nav-active .stButton > button {{
        background: {active_bg} !important;
        color: #000 !important;
        font-weight: 700 !important;
        box-shadow: 0 3px 12px rgba(0,208,132,0.3) !important;
    }}
    /* ── Icon buttons (notif + avatar) ──────────────── */
    .navbar-row .nav-icon .stButton > button {{
        padding: 7px 12px !important;
        font-size: 16px !important;
        background: rgba(255,255,255,0.04) !important;
        color: #94A3B8 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px !important;
    }}
    .navbar-row .nav-icon .stButton > button:hover {{
        background: rgba(255,255,255,0.08) !important;
        color: white !important;
        transform: none !important;
        box-shadow: none !important;
    }}
    /* ── Avatar button: green ────────────────────────── */
    .navbar-row .nav-avatar .stButton > button {{
        background: linear-gradient(135deg, #00D084, #00A368) !important;
        color: #000 !important;
        font-weight: 800 !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(0,208,132,0.3) !important;
    }}
    /* ── Brand text (HTML inside column) ────────────── */
    .nav-brand {{
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
        padding: 0 8px;
    }}
    .nav-brand-dot {{
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #00D084;
        box-shadow: 0 0 8px #00D084;
        animation: pulse-dot 2s infinite;
        flex-shrink: 0;
    }}
    .nav-brand-text {{
        color: #00D084;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: -0.02em;
        font-family: 'DM Sans', sans-serif;
    }}
    /* ── Notif popup ─────────────────────────────────── */
    .sp-notif-popup {{
        position: fixed;
        top: 72px;
        right: 24px;
        width: 300px;
        background: #0D1625;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 18px;
        padding: 14px 16px 8px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(0,208,132,0.06);
        z-index: 9999;
        animation: notif-in 0.16s cubic-bezier(0.2,0,0,1);
    }}
    @keyframes notif-in {{
        from {{ opacity:0; transform: translateY(-10px) scale(0.96); }}
        to   {{ opacity:1; transform: translateY(0) scale(1); }}
    }}
    .sp-notif-title {{
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 10px;
        font-family: 'DM Sans', sans-serif;
    }}
    .sp-notif-item {{
        display: flex;
        align-items: center;
        gap: 10px;
        color: #CBD5E1;
        font-size: 13px;
        padding: 9px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-family: 'DM Sans', sans-serif;
        line-height: 1.4;
    }}
    .sp-notif-item:last-child {{ border-bottom: none; }}
    </style>
    """
    st.markdown(btn_css, unsafe_allow_html=True)

    # ── Render the single navbar row ──────────────────────────────────────────
    st.markdown('<div class="navbar-row">', unsafe_allow_html=True)

    # Columns: brand | 4 nav tabs | spacer | notif | avatar
    cols = st.columns([1.6, 0.8, 0.9, 0.7, 1.1, 0.2, 0.45, 0.45])

    with cols[0]:
        st.markdown('<div class="nav-brand"><div class="nav-brand-dot"></div><span class="nav-brand-text">StockPredict AI</span></div>', unsafe_allow_html=True)

    for i, tab in enumerate(NAV_TABS):
        with cols[i + 1]:
            css_cls = "nav-active" if tab == active else "nav-tab"
            st.markdown(f'<div class="{css_cls}">', unsafe_allow_html=True)
            if st.button(tab, key=f"nav_{tab}", use_container_width=True):
                st.session_state.nav_tab = tab
                st.session_state.notif_open = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # spacer col[5] intentionally empty

    with cols[6]:
        notif_icon = "🔔" if not notif_open else "🔕"
        st.markdown('<div class="nav-icon">', unsafe_allow_html=True)
        if st.button(notif_icon, key="nav_notif", use_container_width=True):
            st.session_state.notif_open = not notif_open
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with cols[7]:
        st.markdown('<div class="nav-avatar">', unsafe_allow_html=True)
        if st.button(avatar_label, key="nav_profile", use_container_width=True):
            st.session_state.nav_tab = "Profile"
            st.session_state.notif_open = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close .navbar-row

    # ── Floating notification popup ───────────────────────────────────────────
    if notif_open:
        notifs = [
            ("🟢", "BBCA.JK naik 1.2% hari ini"),
            ("🔴", "GOTO.JK turun melewati support"),
            ("🔵", "Model TLKM.JK siap dipakai"),
            ("🟡", "Peringatan: volatilitas tinggi TSLA"),
        ]
        rows_html = "".join(
            f'<div class="sp-notif-item">{icon} <span>{msg}</span></div>'
            for icon, msg in notifs
        )
        st.markdown(f"""
        <div class="sp-notif-popup">
            <div class="sp-notif-title">Notifikasi</div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)


def _stock_header(df_raw, ticker):
    last_close = df_raw["Close"].iloc[-1]
    prev_close = df_raw["Close"].iloc[-2]
    change = last_close - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
    change_class = "sp-change-up" if change >= 0 else "sp-change-down"
    arrow = "↗" if change >= 0 else "↘"

    st.markdown(f"""
    <div class="sp-card" style="margin-bottom:0; border-bottom-left-radius:0; border-bottom-right-radius:0; border-bottom:none;">
        <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:16px;">
            <div>
                <div class="sp-stock-name">Selected Stock</div>
                <div class="sp-price">{_fmt_price(last_close, ticker)}</div>
                <div class="{change_class}">{arrow}&nbsp;{_fmt_price(abs(change), ticker)}&nbsp;<span style="opacity:0.7;">({change_pct:+.2f}%) today</span></div>
            </div>
            <div style="text-align:right;">
                <div class="sp-small-label">Ticker</div>
                <div style="font-weight:700; font-size:20px; color:white; font-family:var(--font-mono); letter-spacing:-0.02em;">{ticker}</div>
                <div class="sp-muted" style="font-size:11px; margin-top:6px; letter-spacing:0.04em; text-transform:uppercase;">Live via Yahoo Finance</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive period filter buttons
    PERIODS = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 365, "All": None}
    if "chart_period" not in st.session_state:
        st.session_state.chart_period = "3M"

    st.markdown("""<style>
    /* Period filter buttons — inactive */
    .period-btn-row button[kind="secondary"],
    .period-btn-row button[data-testid="baseButton-secondary"] {
        background: #1a2942 !important;
        color: #94A3B8 !important;
        border: 1px solid #1F2A3D !important;
        border-radius: 10px !important;
        padding: 0.45rem 0.5rem !important;
        font-size: 13px !important;
        font-weight: 700 !important;
        width: 100% !important;
    }
    .period-btn-row button[kind="secondary"]:hover,
    .period-btn-row button[data-testid="baseButton-secondary"]:hover {
        background: #243350 !important;
        color: white !important;
    }
    </style>""", unsafe_allow_html=True)

    st.markdown('<div class="period-btn-row">', unsafe_allow_html=True)
    btn_cols = st.columns(len(PERIODS))
    for col, (label, days) in zip(btn_cols, PERIODS.items()):
        with col:
            if st.session_state.chart_period == label:
                st.markdown(
                    f"<div style='background:#10B981;color:white;border-radius:10px;"
                    f"padding:7px 2px;font-size:13px;font-weight:700;text-align:center;'>{label}</div>",
                    unsafe_allow_html=True
                )
            else:
                if st.button(label, key=f"period_{label}_{ticker}"):
                    st.session_state.chart_period = label
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Filter chart data
    chart_df = df_raw.copy()
    chart_df["Date_str"] = chart_df["Date"].astype(str)
    chart_df["Date_dt"] = pd.to_datetime(chart_df["Date_str"], format="%Y%m%d")
    chart_df = chart_df.set_index("Date_dt")
    selected_days = PERIODS[st.session_state.chart_period]
    if selected_days is not None:
        chart_df = chart_df.tail(selected_days)

    # Build Plotly chart styled like screenshot
    dates = chart_df.index.tolist()
    closes = chart_df["Close"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=closes,
        mode="lines",
        line=dict(color="#00D084", width=2, shape="spline", smoothing=1.2),
        fill="tozeroy",
        fillgradient=dict(
            type="vertical",
            colorscale=[[0, "rgba(0,208,132,0.25)"], [1, "rgba(0,208,132,0.0)"]],
        ),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Harga: <b>%{y:,.2f}</b><extra></extra>",
    ))

    y_min = min(closes) * 0.997
    y_max = max(closes) * 1.003

    fig.update_layout(
        height=290,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#64748B", size=10, family="DM Sans"),
            tickformat="%b %d",
            showline=False,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
            tickfont=dict(color="#64748B", size=10, family="Space Mono"),
            tickformat=",.0f",
            showline=False,
            side="right",
            range=[y_min, y_max],
        ),
        hoverlabel=dict(
            bgcolor="#0D1625",
            bordercolor="#00D084",
            font=dict(color="white", size=12, family="DM Sans"),
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    high_period = chart_df["High"].max()
    low_period  = chart_df["Low"].min()
    avg_volume  = chart_df["Volume"].mean()

    s1, s2, s3, s4 = st.columns(4)
    for col, label, val in [
        (s1, "Period Low", _fmt_price(low_period, ticker)),
        (s2, "Period High", _fmt_price(high_period, ticker)),
        (s3, "Data Points", str(len(chart_df))),
        (s4, "Avg Volume", _fmt_volume(avg_volume)),
    ]:
        with col:
            st.markdown(f"<div class='sp-card-soft'><div class='sp-small-label'>{label}</div><div class='sp-stat-value'>{val}</div></div>", unsafe_allow_html=True)

    return last_close, prev_close, change, change_pct


def _right_panel(ticker, change_pct):
    signal, reason = _signal_from_change(change_pct)
    badge_class = "sp-badge-green" if signal == "BUY" else "sp-badge"
    rec_color_class = "sp-rec-buy" if signal == "BUY" else ("sp-rec-sell" if signal == "SELL" else "sp-rec-hold")
    signal_icon = "↑" if signal == "BUY" else ("↓" if signal == "SELL" else "→")
    st.markdown(f"""
    <div class="sp-card" style="height:100%;">
        <div class="sp-section-label">AI Signal</div>
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px;">
            <div class="sp-rec-title {rec_color_class}">{signal_icon} {signal}</div>
            <div class="{badge_class}">ML Based</div>
        </div>
        <div style="font-weight:600; margin-bottom:8px; font-size:13px; color:var(--sp-muted-light);">Analysis</div>
        <div class="sp-insight-text" style="margin-bottom:20px;">{reason}</div>
        <div class="sp-disclaimer">⚠ Prediction output is for academic purposes only. Not financial advice.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sp-card" style="margin-top:20px;">
        <div class="sp-section-label">Market Intelligence</div>
        <div class="sp-insight">
            <div class="sp-insight-label sp-insight-label-green">Good to know</div>
            <div class="sp-insight-text">Chart uses Yahoo Finance historical data for {ticker}. Prices shown in local currency.</div>
        </div>
        <div class="sp-insight">
            <div class="sp-insight-label sp-insight-label-amber">Model note</div>
            <div class="sp-insight-text">Train the model once and predictions load instantly. Saved models persist between sessions.</div>
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
    st.markdown('<div class="sp-section-title" style="margin-top:32px; margin-bottom:14px;">Watchlist</div>', unsafe_allow_html=True)
    for symbol, name, change, positive in stocks:
        cls = "sp-change-up" if positive else "sp-change-down"
        st.markdown(f"""
        <div class="sp-watch-row">
            <div class="sp-flex">
                <div class="sp-symbol-circle">{symbol[0]}</div>
                <div>
                    <div style="font-weight:700; color:white; font-size:14px; letter-spacing:-0.01em;">{symbol}</div>
                    <div class="sp-muted" style="font-size:12px;">{name}</div>
                </div>
            </div>
            <div class="{cls}" style="margin-top:0; font-family:var(--font-mono); font-size:13px;">{change}</div>
        </div>
        """, unsafe_allow_html=True)


def _page_watchlist():
    st.markdown('<div class="sp-section-title">📋 Watchlist</div>', unsafe_allow_html=True)
    watchlist = [
        ("BBCA.JK", "Bank Central Asia", "Rp 9,250", "+0.85%", True),
        ("TLKM.JK", "Telkom Indonesia", "Rp 3,820", "+0.42%", True),
        ("GOTO.JK", "GoTo Gojek Tokopedia", "Rp 62", "-0.21%", False),
        ("BMRI.JK", "Bank Mandiri", "Rp 6,280", "+0.58%", True),
        ("AAPL",    "Apple Inc.",          "$198.50",  "+1.12%", True),
        ("TSLA",    "Tesla Inc.",           "$175.30",  "-2.30%", False),
    ]
    for sym, name, price, chg, up in watchlist:
        cls = "sp-change-up" if up else "sp-change-down"
        st.markdown(f"""
        <div class="sp-card" style="margin-bottom:12px; padding:16px 20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div class="sp-avatar" style="font-size:13px;">{sym[0]}</div>
                    <div>
                        <div style="font-weight:800; color:white; font-size:15px;">{sym}</div>
                        <div class="sp-muted" style="font-size:12px;">{name}</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-weight:800; color:white; font-size:15px;">{price}</div>
                    <div class="{cls}" style="font-size:13px;">{chg}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _page_alerts():
    st.markdown('<div class="sp-section-title">🔔 Price Alerts</div>', unsafe_allow_html=True)
    if "alerts" not in st.session_state:
        st.session_state.alerts = []

    with st.form("alert_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([1.5, 2, 1.5, 1])
        with c1:
            at = st.text_input("Ticker", placeholder="GOTO.JK")
        with c2:
            ap = st.number_input("Harga Target", min_value=0.0, step=0.5, format="%.2f")
        with c3:
            atype = st.selectbox("Kondisi", ["Di atas", "Di bawah"])
        with c4:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("➕ Tambah")
        if submitted and at:
            st.session_state.alerts.append({"ticker": at.upper(), "price": ap, "type": atype})
            st.success(f"Alert ditambahkan: {at.upper()} {atype} {ap:.2f}")

    if not st.session_state.alerts:
        st.info("Belum ada alert. Tambahkan di atas.")
    else:
        for i, alert in enumerate(st.session_state.alerts):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <div class="sp-card" style="padding:14px 18px; margin-bottom:8px;">
                    <span style="font-weight:800; color:white;">{alert["ticker"]}</span>
                    <span class="sp-muted"> — {alert["type"]} </span>
                    <span style="color:var(--sp-green); font-weight:700;">{alert["price"]:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_alert_{i}"):
                    st.session_state.alerts.pop(i)
                    st.rerun()


def _page_ai_insights():
    st.markdown('<div class="sp-section-title">🤖 AI Insights</div>', unsafe_allow_html=True)
    insights = [
        ("Momentum", "BBCA.JK menunjukkan momentum bullish kuat dengan volume di atas rata-rata 5 hari terakhir.", "sp-insight-label"),
        ("Volatilitas", "GOTO.JK memiliki volatilitas tinggi. Gunakan stop-loss ketat jika masuk posisi.", "sp-insight-label"),
        ("Trend Sektor", "Sektor perbankan Indonesia secara umum sedang dalam uptrend jangka menengah.", "sp-insight-label"),
        ("Risiko Global", "Pasar global sedang dalam mode risk-off. Pantau data inflasi AS minggu ini.", "sp-insight-label"),
        ("Rekomendasi", "Diversifikasi portofolio dengan campuran saham defensive dan growth.", "sp-insight-label"),
    ]
    for title, text, lbl_cls in insights:
        st.markdown(f"""
        <div class="sp-card" style="margin-bottom:14px;">
            <div class="{lbl_cls}" style="margin-bottom:8px;">{title}</div>
            <div class="sp-insight-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)


def _page_profile():
    # ── Extra CSS for login/profile page ─────────────────────────────────────
    st.markdown("""
    <style>
    .login-wrap {
        max-width: 420px;
        margin: 40px auto 0;
    }
    .login-card {
        background: #0D1625;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 36px 32px 28px;
        box-shadow: 0 8px 48px rgba(0,0,0,0.4);
    }
    .login-logo-wrap {
        text-align: center;
        margin-bottom: 28px;
    }
    .login-logo-dot {
        display: inline-block;
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #00D084, #00A368);
        border-radius: 14px;
        line-height: 48px;
        font-size: 22px;
        margin-bottom: 14px;
        box-shadow: 0 6px 24px rgba(0,208,132,0.35);
    }
    .login-title {
        color: white;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0 0 6px;
        font-family: 'DM Sans', sans-serif;
    }
    .login-sub {
        color: #64748B;
        font-size: 13.5px;
        font-family: 'DM Sans', sans-serif;
    }
    .login-divider {
        height: 1px;
        background: rgba(255,255,255,0.06);
        margin: 22px 0;
    }
    .profile-card {
        background: #0D1625;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        overflow: hidden;
        max-width: 480px;
        margin: 0 auto;
        box-shadow: 0 8px 48px rgba(0,0,0,0.4);
    }
    .profile-banner {
        background: linear-gradient(135deg, #00D084 0%, #00A368 100%);
        height: 90px;
        position: relative;
    }
    .profile-avatar-wrap {
        position: absolute;
        bottom: -32px;
        left: 50%;
        transform: translateX(-50%);
    }
    .profile-avatar-big {
        width: 64px; height: 64px;
        background: #0D1625;
        border: 3px solid #0D1625;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 800;
        color: #00D084;
        font-family: 'DM Sans', sans-serif;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .profile-body {
        padding: 44px 28px 24px;
        text-align: center;
    }
    .profile-name {
        font-size: 20px;
        font-weight: 700;
        color: white;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        font-family: 'DM Sans', sans-serif;
    }
    .profile-plan-badge {
        display: inline-block;
        background: rgba(0,208,132,0.12);
        color: #00D084;
        border: 1px solid rgba(0,208,132,0.25);
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 6px;
        font-family: 'DM Sans', sans-serif;
    }
    .profile-stats {
        display: flex;
        gap: 1px;
        background: rgba(255,255,255,0.06);
        border-radius: 14px;
        overflow: hidden;
        margin: 20px 0;
    }
    .profile-stat {
        flex: 1;
        background: #131E30;
        padding: 14px 8px;
        text-align: center;
    }
    .profile-stat-val {
        color: white;
        font-size: 18px;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        letter-spacing: -0.02em;
    }
    .profile-stat-lbl {
        color: #64748B;
        font-size: 11px;
        font-weight: 500;
        margin-top: 3px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-family: 'DM Sans', sans-serif;
    }
    .profile-info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 11px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-family: 'DM Sans', sans-serif;
    }
    .profile-info-row:last-child { border-bottom: none; }
    .profile-info-key { color: #64748B; font-size: 13px; }
    .profile-info-val { color: #E2E8F0; font-size: 13px; font-weight: 500; }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.get("logged_in"):
        user = st.session_state.login_user
        initials = user[:2].upper()

        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-banner">
                <div class="profile-avatar-wrap">
                    <div class="profile-avatar-big">{initials}</div>
                </div>
            </div>
            <div class="profile-body">
                <div class="profile-name">{user}</div>
                <div class="profile-plan-badge">⚡ Pro Plan</div>
                <div class="profile-stats">
                    <div class="profile-stat">
                        <div class="profile-stat-val">3</div>
                        <div class="profile-stat-lbl">Models</div>
                    </div>
                    <div class="profile-stat">
                        <div class="profile-stat-val">12</div>
                        <div class="profile-stat-lbl">Predictions</div>
                    </div>
                    <div class="profile-stat">
                        <div class="profile-stat-val">4</div>
                        <div class="profile-stat-lbl">Watchlist</div>
                    </div>
                </div>
                <div class="profile-info-row">
                    <span class="profile-info-key">Email</span>
                    <span class="profile-info-val">{user.lower()}@stockpredict.ai</span>
                </div>
                <div class="profile-info-row">
                    <span class="profile-info-key">Status</span>
                    <span class="profile-info-val" style="color:#00D084;">● Aktif</span>
                </div>
                <div class="profile-info-row">
                    <span class="profile-info-key">Member sejak</span>
                    <span class="profile-info-val">Juni 2025</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
            if st.button("Logout", key="logout_btn", use_container_width=True):
                st.session_state.logged_in  = False
                st.session_state.login_user = ""
                st.session_state.nav_tab    = "Home"
                st.rerun()

    else:
        # ── Login page ────────────────────────────────────────────────────────
        st.markdown("""
        <div class="login-wrap">
            <div class="login-logo-wrap">
                <div class="login-logo-dot">📈</div>
                <div class="login-title">Masuk ke StockPredict AI</div>
                <div class="login-sub">Pantau saham & lihat prediksi AI kamu</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        _, mid, _ = st.columns([1, 2.2, 1])
        with mid:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username / Email", placeholder="contoh: john@email.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                col_chk, col_spacer = st.columns([1, 1])
                with col_chk:
                    st.checkbox("Ingat saya")
                submitted = st.form_submit_button("Masuk", use_container_width=True)
                if submitted:
                    if username and password:
                        st.session_state.logged_in  = True
                        st.session_state.login_user = username.split("@")[0].capitalize()
                        st.success(f"Selamat datang, {st.session_state.login_user}! 🎉")
                        st.rerun()
                    else:
                        st.error("Username dan password wajib diisi.")

        st.markdown('<div style="text-align:center; color:#64748B; font-size:13px; margin-top:18px;">Belum punya akun? <a href="#" style="color:#00D084; font-weight:600;">Daftar gratis</a></div>', unsafe_allow_html=True)


def main():
    _top_nav()

    active_tab = st.session_state.get("nav_tab", "Home")

    if active_tab == "Watchlist":
        _page_watchlist()
        return
    elif active_tab == "Alerts":
        _page_alerts()
        return
    elif active_tab == "AI Insights":
        _page_ai_insights()
        return
    elif active_tab == "Profile":
        _page_profile()
        return

    # ── HOME tab ──
    with st.sidebar:
        st.markdown("### Configuration")
        ticker = st.text_input(
            "Stock Ticker Symbol",
            value="GOTO.JK",
            placeholder="Example: GOTO.JK, BBCA.JK, AAPL",
            help="Indonesian stocks use .JK suffix. Example: BBCA.JK"
        ).upper().strip()

        epochs_train = st.slider("Training Epochs", 20, 100, 100, 10)
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
        # Auto-scroll to forecast section
        components.html("""
        <script>
            window.parent.document.getElementById('forecast-anchor').scrollIntoView({behavior: 'smooth', block: 'start'});
        </script>
        """, height=0)
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
            width='stretch',
        )


def _run_training(ticker, df_raw, epochs_train, epochs_finetune):
    """Train, save, and then predict."""
    import tensorflow as tf

    st.markdown('<div class="section-header"><div class="section-header-dot"></div><h3>Model Training</h3></div>', unsafe_allow_html=True)

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
    # Anchor for auto-scroll
    st.markdown('<div id="forecast-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><div class="section-header-dot"></div><h3>5-Day Price Forecast</h3></div>', unsafe_allow_html=True)

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
        📌 Reference &mdash; Last Close: <strong style="color:white; font-family:var(--font-mono);">{last_close:.2f}</strong> &nbsp;|&nbsp;
        Last Mid-Price (High+Low)/2: <strong style="color:white; font-family:var(--font-mono);">{last_mid:.2f}</strong><br>
        <span style="font-size:12px; margin-top:4px; display:block;">Predictions represent estimated mid-price for each of the next 5 trading days.</span>
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
    st.dataframe(summary.set_index("Day"), width='stretch')

    st.markdown("""
    <div class="sp-disclaimer" style="font-size:12.5px; margin-top:6px;">
        ⚠️ <strong>Disclaimer:</strong> Predictions are generated by a machine learning model and are for
        educational purposes only. Past performance does not guarantee future results.
        Always consult a qualified financial professional before making investment decisions.
    </div>
    """, unsafe_allow_html=True)




if __name__ == "__main__":
    main()