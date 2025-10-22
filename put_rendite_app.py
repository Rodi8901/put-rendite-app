import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from math import log, sqrt
from scipy.stats import norm

# === THEORETISCHES DELTA (PUT, Black-Scholes-Näherung) ===
def put_delta(S, K, T, r, sigma):
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        return norm.cdf(d1) - 1
    except:
        return np.nan

# === SEITENTITEL ===
st.set_page_config(page_title="Put-Rendite-Rechner", layout="wide")
st.title("💰 Put-Rendite-Rechner – Version 7.6")
st.caption("Berechnet Rendite, Delta, Sicherheitsabstand und ±1σ-Bereich basierend auf Yahoo Finance-Daten.")

# === EINGABEN ===
ticker = st.text_input("Ticker (z. B. AAPL, INTC, NVDA):", "INTC").upper()
strikes_input = st.text_input("Strikes (Komma-getrennt):", "31, 32, 33")
show_all = st.checkbox("Alle Laufzeiten anzeigen (nicht nur 25–60 Tage)", False)

fee_per_option = st.number_input("Gebühr pro Option ($):", min_value=0.0, max_value=50.0, value=3.5, step=0.5)
options_per_trade = st.number_input("Anzahl gehandelter Optionen:", min_value=1, max_value=50, value=1, step=1)
risk_free_rate = st.number_input("Risikofreier Zins (z. B. 0.05 für 5 %):", min_value=0.0, max_value=0.2, value=0.05, step=0.01)

# === BUTTON: START ===
if st.button("📊 Renditen berechnen"):
    try:
        strikes = [float(s) for s in strikes_input.split(",") if s.strip()]
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]
        opt_dates = stock.options
        today = datetime.now()
        rows = []

        atm_iv = None  # ATM-IV für σ-Bereich

        for exp in opt_dates:
            chain = stock.option_chain(exp)
            puts = chain.puts

            # ATM-IV (Strike am nächsten am Kurs)
            if atm_iv is None and not puts.empty:
                puts["diff"] = abs(puts["strike"] - current_price)
                closest = puts.loc[puts["diff"].idxmin()]
                atm_iv = closest["impliedVolatility"]
                expiration = datetime.strptime(exp, "%Y-%m-%d")
                days_to_exp = (expiration - today).days
                T = days_to_exp / 365
                sigma_abs = current_price * atm_iv * sqrt(T)

                st.markdown(f"### 📈 At-the-Money-Volatilität (ATM IV): {atm_iv*100:.2f} %")
                st.markdown(f"**Erwartete Standardabweichung (±1σ): ±{sigma_abs:.2f} $ → ({current_price - sigma_abs:.2f} $ – {current_price + sigma_abs:.2f} $ bis Verfall)**")

            for strike in strikes:
                row = puts[puts["strike"] == strike]
                if not row.empty:
                    bid = float(row["bid"].iloc[0])
                    iv = float(row["impliedVolatility"].iloc[0])
                    expiration = datetime.strptime(exp, "%Y-%m-%d")
                    days = (expiration - today).days
                    T = days / 365
                    sigma = iv if not np.isnan(iv) else 0.3

                    # Delta
                    delta = put_delta(current_price, strike, T, risk_free_rate, sigma)

                    # Rendite
                    total_premium = bid * 100 * options_per_trade
                    total_fee = fee_per_option * options_per_trade
                    net_premium = total_premium - total_fee
                    capital = strike * 100 * options_per_trade

                    if capital > 0 and days > 0:
                        roi_trade = net_premium / capital
                        annualized_roi = roi_trade * (365 / days)
                    else:
                        roi_trade, annualized_roi = np.nan, np.nan

                    # Sicherheit
                    safety = ((strike - current_price) / current_price) * 100

                    # ±1σ
                    move = current_price * sigma * sqrt(T)
                    lower = current_price - move
                    upper = current_price + move

                    rows.append({
                        "Laufzeit (Tage)": days,
                        "Strike": strike,
                        "Bid ($)": bid,
                        "IV (%)": sigma * 100,
                        "Delta": delta,
                        "Sicherheit (%)": safety,
                        "Netto Prämie ($)": net_premium,
                        "Kapital ($)": capital,
                        "Rendite Trade (%)": roi_trade * 100,
                        "Rendite p.a. (%)": annualized_roi * 100,
                        "±1σ Untergrenze ($)": lower,
                        "±1σ Obergrenze ($)": upper
                    })

        df = pd.DataFrame(rows)
        if not show_all:
            df = df[(df["Laufzeit (Tage)"] >= 25) & (df["Laufzeit (Tage)"] <= 60)]

        if df.empty:
            st.warning("Keine passenden Optionen mit 25–60 Tagen Laufzeit gefunden.")
        else:
            df = df.sort_values(by=["Laufzeit (Tage)", "Strike"])

            # Rundung aller Zahlen auf 2 Nachkommastellen
            float_cols = df.select_dtypes(include=["float64"]).columns
            df[float_cols] = df[float_cols].round(2)

            # Style-Definition: Rendite fett + grün
            def highlight_roi(row):
                style = [''] * len(row)
                if "Rendite p.a. (%)" in df.columns:
                    style[df.columns.get_loc("Rendite p.a. (%)")] = "font-weight: bold; color: #006400"
                return style

            # Trenner zwischen Laufzeiten
            styled_df = (
                df.style
                .apply(highlight_roi, axis=1)
                .set_table_styles([
                    {"selector": "thead th", "props": [("font-weight", "bold"), ("border-bottom", "2px solid black")]},
                    {"selector": "tbody tr", "props": [("border-top", "1px solid #ccc")]},
                ])
                .format(precision=2)
            )

            st.subheader(f"📋 Ergebnisse für {ticker} (Kurs: {current_price:.2f} $)")
            st.dataframe(styled_df, use_container_width=True)

            # Diagramm
            plt.figure(figsize=(10, 6))
            for strike in strikes:
                subset = df[df["Strike"] == strike]
                if not subset.empty:
                    plt.plot(subset["Laufzeit (Tage)"], subset["Rendite p.a. (%)"],
                             marker="o", label=f"Strike {strike}$")

            plt.title(f"Put-Rendite p.a. für {ticker}")
            plt.xlabel("Laufzeit (Tage)")
            plt.ylabel("Rendite p.a. (%)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Fehler: {e}")
