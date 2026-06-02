"""
Stock Price Prediction - Streamlit App
Converted from Jupyter Notebook with model persistence (train once, predict forever)
"""

import streamlit as st
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

# ─── CSS Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* Main header */
.hero-header {
    background: linear-gradient(135deg, #1a2744 0%, #0f3460 50%, #16213e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center, rgba(88,166,255,0.05) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5d6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-subtitle {
    font-size: 1rem;
    color: #8b949e;
    margin-top: 0.4rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    border-color: #58a6ff;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(88,166,255,0.15);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #58a6ff, #79c0ff);
}
.metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
.metric-sub { font-size: 0.8rem; color: #8b949e; margin-top: 0.2rem; }

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #30363d;
}
.section-header h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #c9d1d9;
    margin: 0;
}

/* Prediction result cards */
.pred-card {
    background: linear-gradient(135deg, #1a2744 0%, #162032 100%);
    border: 1px solid #1f6feb;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
}
.pred-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(31,111,235,0.25);
}
.pred-day { font-size: 0.7rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; }
.pred-price { font-size: 1.5rem; font-weight: 700; color: #79c0ff; }
.pred-change { font-size: 0.8rem; margin-top: 0.2rem; }
.pred-up { color: #3fb950; }
.pred-down { color: #f85149; }

/* Status badges */
.badge-success { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.75rem; display: inline-block; }
.badge-info { background: rgba(88,166,255,0.15); color: #58a6ff; border: 1px solid rgba(88,166,255,0.3); border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.75rem; display: inline-block; }
.badge-warning { background: rgba(210,153,34,0.15); color: #d2a123; border: 1px solid rgba(210,153,34,0.3); border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.75rem; display: inline-block; }

/* Info box */
.info-box {
    background: rgba(88,166,255,0.07);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: #c9d1d9;
    line-height: 1.6;
}

/* Table styling */
.stDataFrame { border: 1px solid #30363d !important; border-radius: 8px !important; }

/* Streamlit overrides */
.stButton>button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}
.stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(31,111,235,0.4) !important;
}
.stTextInput>div>div>input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}
.stSelectbox>div>div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
}
.stProgress .st-bo { background: #1f6feb !important; }
div[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
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
        future_cols[f"Target+{i + 1}"] = df_temp["Target"].shift(periods=-(i))
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
    for lag in reversed(range(PAST_DAYS_LAG)):   # -19 to -0
        row = recent[lag]
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
def main():
    # Hero header
    st.markdown("""
    <div class="hero-header">
        <p class="hero-title">📈 Stock Price Predictor</p>
        <p class="hero-subtitle">AI-powered 5-day stock price forecast using deep learning (Conv1D neural network)</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        ticker = st.text_input(
            "Stock Ticker Symbol",
            value="GOTO.JK",
            placeholder="e.g. GOTO.JK, BBCA.JK, AAPL",
            help="Indonesian stocks: add .JK suffix (e.g. GOTO.JK)\nUS stocks: just the symbol (e.g. AAPL)",
        ).upper().strip()

        epochs_train = st.slider("Training Epochs (1st pass)", min_value=20, max_value=200, value=100, step=10)
        epochs_finetune = st.slider("Fine-tune Epochs (2nd pass)", min_value=5, max_value=50, value=10, step=5)

        st.markdown("---")
        st.markdown("### 🗂️ Model Cache")

        if ticker:
            if model_exists(ticker):
                st.markdown(f'<span class="badge-success">✓ Saved model found for {ticker}</span>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                    Model is <strong>already trained</strong> and saved on disk.
                    Click <em>Predict</em> to get instant results — no retraining needed!
                </div>
                """, unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    btn_predict = st.button("⚡ Predict Now", key="predict_saved")
                with col_b:
                    btn_retrain = st.button("🔄 Retrain", key="retrain")
            else:
                st.markdown(f'<span class="badge-warning">⚠ No model for {ticker}</span>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                    No saved model found. Click <em>Train & Predict</em> to download data,
                    train, save, and predict. Future uses will be instant!
                </div>
                """, unsafe_allow_html=True)
                btn_predict = False
                btn_retrain = False
                btn_train = st.button("🚀 Train & Predict", key="train_new")

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem; color:#8b949e; line-height:1.8;">
        <strong style="color:#c9d1d9;">How it works:</strong><br>
        1. Downloads stock data (last 2 years)<br>
        2. Applies outlier treatment<br>
        3. Creates 20-day lookback features<br>
        4. Trains Conv1D neural network<br>
        5. Saves model to disk<br>
        6. Future predictions: instant! ⚡
        </div>
        """, unsafe_allow_html=True)

    # ── Main Content ─────────────────────────────────────────────────────────
    if not ticker:
        st.info("👈 Enter a stock ticker in the sidebar to get started.")
        return

    # Determine which action to take
    action = None
    if model_exists(ticker):
        if "btn_predict" in dir() and btn_predict:
            action = "predict"
        elif "btn_retrain" in dir() and btn_retrain:
            action = "retrain"
    else:
        if "btn_train" in dir() and btn_train:
            action = "train"

    # ── Show current stock info ──────────────────────────────────────────────
    try:
        with st.spinner(f"Loading data for {ticker}…"):
            df_raw = download_stock_data(ticker)

        # Summary metrics
        st.markdown('<div class="section-header"><h3>📊 Market Overview</h3></div>', unsafe_allow_html=True)
        last_close = df_raw["Close"].iloc[-1]
        prev_close = df_raw["Close"].iloc[-2]
        change = last_close - prev_close
        change_pct = (change / prev_close) * 100
        high52 = df_raw["High"].max()
        low52  = df_raw["Low"].min()
        avg_vol = df_raw["Volume"].mean()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            arrow = "▲" if change >= 0 else "▼"
            color = "#3fb950" if change >= 0 else "#f85149"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Last Close</div>
                <div class="metric-value">{last_close:.2f}</div>
                <div class="metric-sub" style="color:{color}">{arrow} {abs(change):.2f} ({change_pct:+.2f}%)</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">52W High</div>
                <div class="metric-value">{high52:.2f}</div>
                <div class="metric-sub">Period Maximum</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">52W Low</div>
                <div class="metric-value">{low52:.2f}</div>
                <div class="metric-sub">Period Minimum</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Data Points</div>
                <div class="metric-value">{len(df_raw)}</div>
                <div class="metric-sub">Trading Days</div>
            </div>""", unsafe_allow_html=True)
        with col5:
            avg_vol_str = f"{avg_vol/1e9:.2f}B" if avg_vol > 1e9 else f"{avg_vol/1e6:.1f}M"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Volume</div>
                <div class="metric-value">{avg_vol_str}</div>
                <div class="metric-sub">Daily Average</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Price chart
        st.markdown('<div class="section-header"><h3>📉 Price History</h3></div>', unsafe_allow_html=True)
        chart_df = df_raw.copy()
        chart_df["Date_str"] = chart_df["Date"].astype(str)
        # Convert back to datetime for chart
        chart_df["Date_dt"] = pd.to_datetime(chart_df["Date_str"], format="%Y%m%d")
        chart_display = chart_df.set_index("Date_dt")[["Close", "High", "Low"]].rename(
            columns={"Close": "Close Price", "High": "Day High", "Low": "Day Low"}
        )
        st.line_chart(chart_display, color=["#58a6ff", "#3fb950", "#f85149"], height=280)

    except Exception as e:
        st.error(f"❌ Error loading data for **{ticker}**: {e}")
        return

    # ── Training / Prediction Logic ──────────────────────────────────────────
    if action == "predict":
        _run_prediction(ticker)

    elif action in ("train", "retrain"):
        _run_training(ticker, df_raw, epochs_train, epochs_finetune)

    elif model_exists(ticker):
        st.markdown("""
        <div class="info-box">
            ✅ A trained model is ready. Click <strong>⚡ Predict Now</strong> in the sidebar to see predictions!
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-box">
            🚀 No model saved yet. Click <strong>🚀 Train & Predict</strong> in the sidebar to train and save the model.
            After the first training, all future predictions will be <strong>instant</strong>!
        </div>
        """, unsafe_allow_html=True)

    # ── Raw data expander ────────────────────────────────────────────────────
    with st.expander("📋 View Raw Data (last 20 rows)"):
        display_df = df_raw.tail(20).copy()
        display_df["Date"] = pd.to_datetime(display_df["Date"].astype(str), format="%Y%m%d")
        st.dataframe(
            display_df.set_index("Date").style.format(
                {"Close": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}",
                 "Open": "{:.2f}", "Volume": "{:,.0f}"}
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
