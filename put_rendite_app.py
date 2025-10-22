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
st.title("💰 Put-Rendite-Rechner – Version 7.5")
st.caption("Berechnet Rendite, Delta, Sicherheitsabstand und ±1σ-Bereich basierend auf Yahoo Finance-Daten.")

# === EINGABEMASKE ===
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

        atm_iv = None  # ATM-IV wird später für σ-Bereich angezeigt

        for exp in opt_dates:
            chain = stock.option_chain(exp)
            puts = chain.puts
            iv_col = "impliedVolatility"

            # ATM-IV berechnen (Strike am nächsten am Kurs)
            if atm_iv is None:
                puts["diff"] = abs(puts["strike"] - current_price)
                closest_strike = puts.loc[puts["diff"].idxmin()]
                atm_iv = closest_strike["impliedVolatility"]
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

                    # Delta berechnen
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

                    # Sicherheitsmarge (Abstand Strike-Kurs)
                    safety = ((strike - current_price) / current_price) * 100

                    # ±1σ-Bereich
                    move = current_price * sigma * sqrt(T)
                    lower = current_price - move
                    upper = current_price + move

                    rows.append({
                        "Laufzeit (Tage)": days,
                        "Strike": strike,
                        "Bid ($)": round(bid, 2),
                        "IV (%)": round(sigma * 100, 2),
                        "Delta": round(delta, 2),
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
            st.warning("Keine passenden Optionen mit 25–60 Tagen Laufzeit gefunden.")
        else:
            # Nach Laufzeit sortieren
            df = df.sort_values(by=["Laufzeit (Tage)", "Strike"])

            # Fett-Markierung für Jahresrendite
            def highlight_roi(row):
                style = [''] * len(row)
                if "Rendite p.a. (%)" in df.columns:
                    style[df.columns.get_loc("Rendite p.a. (%)")] = "font-weight: bold; color: #006400"
                return style

            # Trenner zwischen Laufzeiten
            df["Laufzeit-Trenner"] = df["Laufzeit (Tage)"].astype(str) + " Tage"

            st.subheader(f"📋 Ergebnisse für {ticker} (Kurs: {current_price:.2f} $)")
            st.dataframe(
                df.style
                .apply(highlight_roi, axis=1)
                .set_table_styles(
                    [{"selector": "tbody tr", "props": [("border-top", "2px solid black")]}]
                )
            )

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
