"""
Streamlit app: Short-term Stock Closing Price Predictor
Single-file app that:
- Fetches daily stock data via yfinance
- Computes technical indicators (SMA, EMA, RSI, MACD, volatility)
- Trains a GradientBoostingRegressor (or LinearRegression) to predict next-day Close
- Shows EDA plots, model metrics, Actual vs Predicted, and a "Predict Next Day" button
- Allows retraining and saving/loading the model

Run: `streamlit run stock_predictor_streamlit.py`
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# technical indicators
try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
except Exception:
    RSIIndicator = None
    MACD = None

import matplotlib.pyplot as plt

MODEL_PATH = "stock_model.joblib"

st.set_page_config(page_title="Stock Close Predictor", layout="wide")
st.title("📈 Short-term Stock Closing Price Predictor")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker (yfinance)", value="INFY.NS")
    start_date = st.date_input("Start date", value=datetime.today() - timedelta(days=365*3))
    end_date = st.date_input("End date", value=datetime.today())
    model_choice = st.selectbox("Model", ["GradientBoosting", "LinearRegression"])
    test_size = st.slider("Test size (fraction)", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
    retrain = st.button("Retrain model")
    load_saved = st.button("Load saved model")
    save_after_train = st.checkbox("Save model after training", value=True)
    st.markdown("---")
    st.write("Indicator options")
    use_rsi = st.checkbox("RSI (14)", value=True)
    use_macd = st.checkbox("MACD", value=True)
    sma_short = st.number_input("SMA short window", min_value=2, max_value=200, value=10)
    sma_long = st.number_input("SMA long window", min_value=2, max_value=400, value=20)

@st.cache_data
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return pd.DataFrame()
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    available_cols = [col for col in expected_cols if col in df.columns]
    df = df[available_cols].copy()
    df.reset_index(inplace=True)
    return df

@st.cache_data
def compute_indicators(df, sma_short=10, sma_long=20, use_rsi=True, use_macd=True):
    data = df.copy()
    # Ensure Close is 1D Series
    close = data['Close'].squeeze()

    data[f'SMA_{sma_short}'] = close.rolling(window=sma_short).mean()
    data[f'SMA_{sma_long}'] = close.rolling(window=sma_long).mean()
    data[f'EMA_{sma_short}'] = close.ewm(span=sma_short, adjust=False).mean()
    data['Returns'] = close.pct_change()
    data['Volatility'] = data['Returns'].rolling(window=10).std()

    if use_rsi and RSIIndicator is not None:
        try:
            data['RSI'] = RSIIndicator(close=close, window=14).rsi()
        except Exception:
            data['RSI'] = np.nan
    else:
        data['RSI'] = np.nan

    if use_macd and MACD is not None:
        try:
            macd = MACD(close=close)
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
        except Exception:
            data['MACD'] = np.nan
            data['MACD_signal'] = np.nan
    else:
        data['MACD'] = np.nan
        data['MACD_signal'] = np.nan

    data.dropna(inplace=True)
    return data


def prepare_features(data, sma_short=10, sma_long=20):
    feature_cols = ['Close', f'SMA_{sma_short}', f'SMA_{sma_long}', f'EMA_{sma_short}', 'Volatility']
    # Add RSI and MACD if present
    if 'RSI' in data.columns:
        feature_cols.append('RSI')
    if 'MACD' in data.columns and 'MACD_signal' in data.columns:
        feature_cols += ['MACD', 'MACD_signal']

    data['Target'] = data['Close'].shift(-1)
    data = data.dropna()

    X = data[feature_cols].copy()
    y = data['Target'].copy()
    return X, y, data


def train_model(X_train, y_train, model_choice='GradientBoosting'):
    if model_choice == 'GradientBoosting':
        model = GradientBoostingRegressor(n_estimators=200)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Main app flow
data = fetch_data(ticker, start_date, end_date)
if data.empty:
    st.error("No data fetched. Check ticker or date range.")
    st.stop()

st.subheader(f"Data for {ticker} from {start_date} to {end_date}")
st.dataframe(data.tail(5))

with st.expander("Show EDA plots"):
    st.write("Close price chart")
    st.line_chart(data.set_index('Date')['Close'])

# Compute indicators
ind_df = compute_indicators(data, sma_short=sma_short, sma_long=sma_long, use_rsi=use_rsi, use_macd=use_macd)

st.subheader("Computed indicators (sample)")
st.dataframe(ind_df.tail(5))

# Prepare features and target
X, y, ind_df = prepare_features(ind_df, sma_short=sma_short, sma_long=sma_long)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=test_size)

# Model loading option
model = None
if load_saved:
    try:
        model = joblib.load(MODEL_PATH)
        st.success("Loaded saved model.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# Retrain
if retrain or model is None:
    with st.spinner("Training model..."):
        model = train_model(X_train, y_train, model_choice=model_choice)
        st.success("Training completed.")
        if save_after_train:
            try:
                joblib.dump(model, MODEL_PATH)
                st.info(f"Model saved to {MODEL_PATH}")
            except Exception as e:
                st.warning(f"Could not save model: {e}")

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

st.subheader("Model performance on test set")
col1, col2, col3 = st.columns(3)
col1.metric("R²", f"{r2:.4f}")
col2.metric("MAE", f"{mae:.4f}")
col3.metric("RMSE", f"{rmse:.4f}")

# Plot Actual vs Predicted
st.subheader("Actual vs Predicted (test)")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(y_test.values, label='Actual')
ax.plot(y_pred_test, label='Predicted')
ax.legend()
st.pyplot(fig)

# Predict next day
st.subheader("Predict next day close")
latest_row = X.iloc[-1:]
pred_next = model.predict(latest_row)[0]
st.write(f"Model predicts next-day closing price: **{pred_next:.2f}**")

# Option to show most recent features used
if st.checkbox("Show latest features used for prediction"):
    st.dataframe(latest_row)

# Download model
if st.button("Download trained model (.joblib)"):
    try:
        with open(MODEL_PATH, 'rb') as f:
            st.download_button(label="Download model", data=f, file_name=MODEL_PATH, mime='application/octet-stream')
    except Exception:
        st.error("Model file not found. Train and save model first.")

st.markdown("---")
st.write("**Notes & next steps:**")
st.write("- This app trains a simple model on historical daily data and predicts next day's close. For production, consider retraining schedule, sliding-window retraining, and more robust feature engineering (news/sentiment, macro features).")
# st.write("- For sequence models (LSTM) you'd need to reshape data and use Keras — can be added as an advanced option.")



# EOF
