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

try:
    fee_per_option = st.number_input(
        "Gebühr pro Option ($):", min_value=0.0, max_value=50.0, value=3.5, step=0.5
    )
except:
    fee_per_option = 3.5

try:
    options_per_trade = st.number_input(
        "Anzahl gehandelter Optionen:", min_value=1, max_value=50, value=1, step=1
    )
except:
    options_per_trade = 1

try:
    risk_free_rate = st.number_input(
        "Risikofreier Zins (z. B. 0.05 für 5 %):", min_value=0.0, max_value=0.2, value=0.05, step=0.01
    )
except:
    risk_free_rate = 0.05

# === BUTTON: START ===
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

            # --- ATM-Strike bestimmen ---
            puts["ATM_Diff"] = abs(puts["strike"] - current_price)
            atm_row = puts.loc[puts["ATM_Diff"].idxmin()]
            atm_strike = atm_row["strike"]
            atm_iv = atm_row["impliedVolatility"] if iv_col else np.nan

            expiration = datetime.strptime(exp, "%Y-%m-%d")
            days = (expiration - today).days
            T = days / 365
            sigma_atm = atm_iv

            # --- ATM-basierte Standardabweichung für Infoanzeige ---
            move = current_price * sigma_atm * sqrt(T)
            lower = current_price - move
            upper = current_price + move

            for strike in strikes:
                row = puts[puts["strike"] == strike]
                if not row.empty:
                    bid = float(row["bid"].iloc[0])
                    iv = float(row[iv_col].iloc[0]) if iv_col else np.nan
                    sigma = iv if not np.isnan(iv) else sigma_atm

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

                    # Sicherheitsmarge
                    safety = ((strike - current_price) / current_price) * 100

                    rows.append({
                        "Strike": strike,
                        "Laufzeit (Tage)": days,
                        "Bid ($)": round(bid, 2),
                        "IV (%)": round(sigma * 100, 2),
                        "Delta": round(delta, 3),
                        "Sicherheit (%)": round(safety, 2),
                        "Netto Prämie ($)": round(net_premium, 2),
                        "Kapital ($)": int(capital),
                        "Rendite Trade (%)": round(roi_trade * 100, 2),
                        "Rendite p.a. (%)": round(annualized_roi * 100, 2)
                    })

            # --- Ausgabe nur für den ersten passenden Verfall (ATM-Daten) ---
            break

        df = pd.DataFrame(rows)
        if not show_all:
            df = df[(df["Laufzeit (Tage)"] >= 25) & (df["Laufzeit (Tage)"] <= 60)]

        if df.empty:
            st.warning("Keine passenden Optionen mit 25–60 Tagen Laufzeit gefunden.")
        else:
            st.subheader(f"📋 Ergebnisse für {ticker} (Kurs: {round(current_price, 2)} $)")
            st.markdown(
                f"**At-the-Money-Volatilität (ATM IV):** {atm_iv:.2%}  \n"
                f"**Erwartete Standardabweichung (±1σ):** ±{move:.2f} $  "
                f"→ ({lower:.2f} $ – {upper:.2f} $ bis Verfall)"
            )

            # Fett-Markierung für Jahresrendite
            def highlight_roi(row):
                style = [''] * len(row)
                if "Rendite p.a. (%)" in df.columns:
                    style[df.columns.get_loc("Rendite p.a. (%)")] = "font-weight: bold; color: #006400"
                return style

            st.dataframe(df.style.apply(highlight_roi, axis=1))

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
