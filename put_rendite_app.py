import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

st.title("💰 Put-Rendite-Rechner")

# Eingaben
ticker = st.text_input("Ticker", "AAPL").upper()
strikes_input = st.text_input("Strikes (kommagetrennt)", "150,155,160")
show_all = st.checkbox("Alle Laufzeiten anzeigen?", False)
fee_per_option = st.number_input("Gebühr pro Option ($)", value=3.5)
options_per_trade = st.number_input("Optionen pro Trade", value=1, step=1)

if st.button("Berechnen"):
    strikes = [float(s.strip()) for s in strikes_input.split(",") if s.strip()]
    stock = yf.Ticker(ticker)
    today = datetime.now()
    current_price = stock.history(period="1d")["Close"].iloc[-1]
    rows = []

    for exp in stock.options:
        chain = stock.option_chain(exp)
        puts = chain.puts
        expiration = datetime.strptime(exp, "%Y-%m-%d")
        days = (expiration - today).days

        for strike in strikes:
            row = puts[puts['strike'] == strike]
            if not row.empty:
                bid = float(row['bid'].iloc[0])
                iv = float(row['impliedVolatility'].iloc[0])
                delta = float(row['delta'].iloc[0])
                if not show_all and not (25 <= days <= 60):
                    continue

                premium = bid * 100 * options_per_trade
                fee = fee_per_option * options_per_trade
                capital = strike * 100 * options_per_trade
                roi_trade = (premium - fee) / capital
                roi_year = roi_trade * (365 / days)

                sigma_move = current_price * iv * np.sqrt(days / 365)
                lower = current_price - sigma_move
                upper = current_price + sigma_move

                rows.append([exp, strike, bid, delta, iv * 100, days,
                             roi_trade * 100, roi_year * 100,
                             sigma_move, lower, upper])

    df = pd.DataFrame(rows, columns=[
        "Verfall", "Strike", "Bid", "Delta", "IV (%)", "Laufzeit (Tage)",
        "Rendite/Trade (%)", "Rendite/Jahr (%)",
        "1σ Move ($)", "1σ Untergrenze ($)", "1σ Obergrenze ($)"
    ])
    st.write(f"**Aktueller Kurs:** {current_price:.2f} $")
    st.dataframe(df)

    # Diagramm
    fig, ax = plt.subplots()
    for strike in strikes:
        subset = df[df["Strike"] == strike]
        ax.plot(subset["Laufzeit (Tage)"], subset["Rendite/Jahr (%)"], marker="o", label=f"Strike {strike}")
    ax.set_xlabel("Laufzeit (Tage)")
    ax.set_ylabel("Rendite p.a. (%)")
    ax.legend()
    st.pyplot(fig)
