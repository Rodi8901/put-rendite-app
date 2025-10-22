import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from math import log, sqrt, exp
from scipy.stats import norm

# === BLACK-SCHOLES DELTA (PUT) ===
def put_delta(S, K, T, r, sigma):
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        return norm.cdf(d1) - 1  # Put-Delta
    except:
        return np.nan

st.title("💰 Put-Rendite-Rechner – Version 7.3")
st.write("Berechnet Rendite, Delta, Sicherheitspuffer und σ-Bereich basierend auf Yahoo Finance-Daten.")

# === Eingabemaske ===
ticker = st.text_input("Ticker (z. B. AAPL, INTC, NVDA):", "INTC").upper()
strikes_input = st.text_input("Strikes (Komma-getrennt):", "31, 32, 33")
show_all = st.checkbox("Alle Laufzeiten anzeigen (nicht nur 25–60 Tage)", False)
fee_per_option = st.number_input("Gebühr pro Option ($):", 3.5, 0.0, 50.0)
options_per_trade = st.number_input("Anzahl gehandelter Optionen:", 1, 1, 50)
risk_free_rate = st.number_input("Risikofreier Zins (z. B. 0.05 für 5 %):", 0.05, 0.0, 0.2)

if st.button("📊 Renditen berechnen"):
    try:
        strikes = [float(s) for s in strikes_input.split(",") if s.strip()]
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]
        opt_dates = stock.options
        today = datetime.now()
        rows = []

        for exp in opt_dates:
            chain = stock.option_chain(exp)
            puts = chain.puts
            iv_col = "impliedVolatility" if "impliedVolatility" in puts.columns else None

            for strike in strikes:
                row = puts[puts["strike"] == strike]
                if not row.empty:
                    bid = float(row["bid"].iloc[0])
                    iv = float(row[iv_col].iloc[0]) if iv_col else np.nan
                    expiration = datetime.strptime(exp, "%Y-%m-%d")
                    days = (expiration - today).days
                    T = days / 365
                    sigma = iv if not np.isnan(iv) else 0.3  # fallback

                    # === Delta (theoretisch) ===
                    delta = put_delta(current_price, strike, T, risk_free_rate, sigma)

                    # === Renditeberechnung ===
                    total_premium = bid * 100 * options_per_trade
                    total_fee = fee_per_option * options_per_trade
                    net_premium = total_premium - total_fee
                    capital = strike * 100 * options_per_trade

                    if capital > 0 and days > 0:
                        roi_trade = net_premium / capital
                        annualized_roi = roi_trade * (365 / days)
                    else:
                        roi_trade = np.nan
                        annualized_roi = np.nan

                    # === Sicherheitsmarge (Strike-Abstand) ===
                    safety = ((strike - current_price) / current_price) * 100

                    # === ±1σ-Bereich ===
                    move = current_price * sigma * sqrt(T)
                    lower = current_price - move
                    upper = current_price + move

                    rows.append({
                        "Strike": strike,
                        "Laufzeit (Tage)": days,
                        "Bid ($)": round(bid, 2),
                        "IV": round(sigma * 100, 2),
                        "Delta": round(delta, 3),
                        "Sicherheit (%)": round(safety, 2),
                        "Netto Prämie ($)": round(net_premium, 2),
                        "Kapital ($)": int(capital),
                        "Rendite Trade (%)": round(roi_trade * 100, 2),
                        "Rendite p.a. (%)": round(annualized_roi * 100, 2),
                        "±1σ Untergrenze ($)": round(lower, 2),
                        "±1σ Obergrenze ($)": round(upper, 2)
                    })

        df = pd.DataFrame(rows)
        if not show_all:
            df = df[(df["Laufzeit (Tage)"] >= 25) & (df["Laufzeit (Tage)"] <= 60)]

        if df.empty:
            st.warning("Keine passenden Optionen gefunden (25–60 Tage).")
        else:
            # === Rendite fett darstellen ===
            def highlight_roi(val):
                return "font-weight: bold" if val.name == "Rendite p.a. (%)" else ""

            st.subheader(f"📋 Ergebnisse für {ticker} (Kurs: {round(current_price, 2)} $)")
            st.dataframe(df.style.apply(highlight_roi))

            # === Diagramm ===
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
        st.error(f"Fehler: {e}")
