# ---
# 💰 Put-Rendite-Rechner – Version 7.1 (Cloud-ready)
# ---
!pip install yfinance pandas numpy matplotlib tabulate streamlit > /dev/null

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

st.title("💰 Put-Rendite-Rechner – Version 7.1")

# === Eingaben ===
ticker = st.text_input("Ticker", "AAPL").upper()
strikes_input = st.text_input("Strikes (kommagetrennt)", "150,155,160")
show_all = st.checkbox("Alle Laufzeiten anzeigen?", False)
fee_per_option = st.number_input("Gebühr pro Option ($)", value=3.5)
options_per_trade = st.number_input("Optionen pro Trade", value=1, step=1)

if st.button("Berechnen"):

    strikes = [float(s.strip()) for s in strikes_input.split(",") if s.strip()]
    stock = yf.Ticker(ticker)
    today = datetime.now()

    try:
        current_price = stock.history(period="1d")["Close"].iloc[-1]
    except:
        st.error("⚠️ Kurs konnte nicht geladen werden.")
        st.stop()

    rows = []

    for exp in stock.options:
        try:
            chain = stock.option_chain(exp)
            puts = chain.puts
            expiration = datetime.strptime(exp, "%Y-%m-%d")
            days_to_exp = (expiration - today).days

            for strike in strikes:
                row = puts[puts['strike'] == strike]
                if row.empty:
                    continue

                bid = float(row['bid'].iloc[0])
                delta = float(row['delta'].iloc[0]) if 'delta' in row.columns else np.nan
                iv = float(row['impliedVolatility'].iloc[0]) if 'impliedVolatility' in row.columns else np.nan

                if not show_all and not (25 <= days_to_exp <= 60):
                    continue

                # Rendite-Berechnung
                premium = bid * 100 * options_per_trade
                fee_total = fee_per_option * options_per_trade
                capital = strike * 100 * options_per_trade
                roi_trade = (premium - fee_total) / capital
                roi_year = roi_trade * (365 / days_to_exp)

                # 1σ-Bereich
                sigma_move = current_price * iv * np.sqrt(days_to_exp / 365) if not np.isnan(iv) else np.nan
                lower_bound = current_price - sigma_move if not np.isnan(sigma_move) else np.nan
                upper_bound = current_price + sigma_move if not np.isnan(sigma_move) else np.nan

                rows.append([
                    exp, strike, bid, delta, iv * 100 if not np.isnan(iv) else np.nan,
                    days_to_exp, roi_trade * 100, roi_year * 100,
                    sigma_move, lower_bound, upper_bound
                ])

        except Exception as e:
            st.warning(f"Fehler bei {exp}: {e}")

    if not rows:
        st.warning("❌ Keine passenden Optionen gefunden.")
        st.stop()

    df = pd.DataFrame(rows, columns=[
        "Verfall", "Strike", "Bid", "Delta", "IV (%)", "Laufzeit (Tage)",
        "Rendite/Trade (%)", "Rendite/Jahr (%)",
        "1σ Move ($)", "1σ Untergrenze ($)", "1σ Obergrenze ($)"
    ]).sort_values("Laufzeit (Tage)")

    st.write(f"**Aktueller Kurs:** {current_price:.2f} $")
    st.dataframe(df)

    # === Grafik ===
    fig, ax = plt.subplots()
    for strike in strikes:
        subset = df[df["Strike"] == strike]
        if not subset.empty:
            ax.plot(subset["Laufzeit (Tage)"], subset["Rendite/Jahr (%)"], marker="o", label=f"Strike {strike}")
    ax.set_xlabel("Laufzeit (Tage)")
    ax.set_ylabel("Rendite p.a. (%)")
    ax.set_title(f"Rendite vs. Laufzeit – {ticker}")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
