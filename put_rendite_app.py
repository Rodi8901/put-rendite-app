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
    # robuste Prüfung der Eingabewerte
    try:
        if not (np.isfinite(S) and np.isfinite(K) and np.isfinite(T) and np.isfinite(sigma)):
            return np.nan
        if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
            return np.nan
        ratio = S / K
        if ratio <= 0:
            return np.nan
        d1 = (log(ratio) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        return norm.cdf(d1) - 1
    except (ValueError, OverflowError) as err:
        return np.nan

# === SEITENTITEL ===
st.set_page_config(page_title="Put-Rendite-Rechner", layout="wide")
st.title("💰 Put-Rendite-Rechner – Robust Version")
st.caption("Robuste Variante: prüft ungültige Werte, überspringt problematische Optionen und zeigt Warnungen.")

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

        # Aktueller Kurs
        hist = stock.history(period="1d")
        if hist.empty:
            st.error("Kursdaten konnten nicht geladen werden.")
            st.stop()
        current_price = hist["Close"].iloc[-1]
        if not np.isfinite(current_price) or current_price <= 0:
            st.error(f"Ungültiger aktueller Kurs: {current_price}")
            st.stop()

        opt_dates = stock.options
        if not opt_dates:
            st.warning("Für dieses Underlying sind keine Optionsdaten verfügbar.")
            st.stop()

        today = datetime.now()
        rows = []
        warnings = []

        # ATM-IV-Ermittlung: wir nehmen die erste Expiration, suchen ATM dort (falls vorhanden)
        atm_iv_global = None
        atm_info_texts = []

        for exp in opt_dates:
            try:
                chain_tmp = stock.option_chain(exp)
                puts_tmp = chain_tmp.puts
                if puts_tmp.empty:
                    continue
                # Bestimme ATM Strike in dieser Laufzeit
                puts_tmp = puts_tmp.assign(diff=(puts_tmp["strike"] - current_price).abs())
                atm_row_tmp = puts_tmp.loc[puts_tmp["diff"].idxmin()]
                if "impliedVolatility" in atm_row_tmp and not pd.isna(atm_row_tmp["impliedVolatility"]):
                    atm_iv_global = float(atm_row_tmp["impliedVolatility"])
                    exp_dt_tmp = datetime.strptime(exp, "%Y-%m-%d")
                    days_tmp = (exp_dt_tmp - today).days
                    if days_tmp > 0:
                        T_tmp = days_tmp / 365
                        sigma_abs_tmp = current_price * atm_iv_global * sqrt(T_tmp)
                        atm_info_texts.append((exp, atm_iv_global, days_tmp, sigma_abs_tmp))
                # wir brechen nicht sofort ab, sammeln Infos; später zeigen wir die erste sinnvolle
            except Exception:
                continue

        # Zeige ATM-Info (erste gültige)
        if atm_info_texts:
            exp0, atm_iv0, days0, sigma_abs0 = atm_info_texts[0]
            st.markdown(f"### 📈 At-the-Money-Volatilität (ATM IV) für nächstes Verfall: {atm_iv0*100:.2f} % (Expiry {exp0})")
            st.markdown(f"**Erwartete Standardabweichung (±1σ): ±{sigma_abs0:.2f} $ → ({current_price - sigma_abs0:.2f} $ – {current_price + sigma_abs0:.2f} $ bis Verfall)**")
        else:
            st.info("ATM-IV konnte nicht zuverlässig bestimmt werden (keine IV-Daten vorhanden).")

        # Hauptloop über Expirations
        for exp in opt_dates:
            try:
                chain = stock.option_chain(exp)
                puts = chain.puts
                if puts.empty:
                    continue

                expiration = datetime.strptime(exp, "%Y-%m-%d")
                days = (expiration - today).days
                if days <= 0:
                    # verfallene oder heute verfallende Optionen überspringen
                    continue
                T = days / 365

                iv_col = "impliedVolatility" if "impliedVolatility" in puts.columns else None

                for strike in strikes:
                    try:
                        row = puts[puts["strike"] == strike]
                        if row.empty:
                            continue

                        # sichere Extraktion
                        bid = row["bid"].iloc[0] if "bid" in row and not pd.isna(row["bid"].iloc[0]) else 0.0
                        iv = row[iv_col].iloc[0] if iv_col and not pd.isna(row[iv_col].iloc[0]) else np.nan

                        # robuster sigma-fallback
                        if not np.isfinite(iv) or iv <= 0:
                            sigma = atm_iv_global if (atm_iv_global and atm_iv_global > 0) else 0.30
                        else:
                            sigma = float(iv)

                        # Validierungen vor Delta-Berechnung
                        S = float(current_price)
                        K = float(strike)
                        if not (np.isfinite(S) and np.isfinite(K) and S > 0 and K > 0):
                            warnings.append(f"Ungültige Preise S={S}, K={K} (Exp {exp}) — übersprungen")
                            continue
                        if not (np.isfinite(sigma) and sigma > 0):
                            warnings.append(f"Ungültige Volatilität sigma={sigma} (Exp {exp}, Strike {strike}) — überschrieben mit 0.3")
                            sigma = 0.30
                        if T <= 0:
                            warnings.append(f"T <= 0 für Exp {exp} — übersprungen")
                            continue

                        # Delta berechnen (geschützt)
                        delta = put_delta(S, K, T, risk_free_rate, sigma)

                        # Renditeberechnung
                        total_premium = float(bid) * 100 * options_per_trade
                        total_fee = fee_per_option * options_per_trade
                        net_premium = total_premium - total_fee
                        capital = K * 100 * options_per_trade

                        if capital > 0 and days > 0:
                            roi_trade = net_premium / capital
                            annualized_roi = roi_trade * (365 / days)
                        else:
                            roi_trade, annualized_roi = np.nan, np.nan

                        # Sicherheit (%)
                        safety = ((K - S) / S) * 100

                        # ±1σ für diesen Strike (optional, wir berechnen trotzdem)
                        move = S * sigma * sqrt(T)
                        lower = S - move
                        upper = S + move

                        rows.append({
                            "Laufzeit (Tage)": days,
                            "Strike": K,
                            "Bid ($)": round(float(bid), 2),
                            "IV (%)": round(sigma * 100, 2),
                            "Delta": round(delta, 2) if np.isfinite(delta) else np.nan,
                            "Sicherheit (%)": round(safety, 2),
                            "Netto Prämie ($)": round(net_premium, 2),
                            "Kapital ($)": int(capital),
                            "Rendite Trade (%)": round(roi_trade * 100, 2) if np.isfinite(roi_trade) else np.nan,
                            "Rendite p.a. (%)": round(annualized_roi * 100, 2) if np.isfinite(annualized_roi) else np.nan,
                            "±1σ Untergrenze ($)": round(lower, 2),
                            "±1σ Obergrenze ($)": round(upper, 2)
                        })
                    except Exception as row_err:
                        warnings.append(f"Fehler bei Exp {exp}, Strike {strike}: {row_err}")
                        continue

            except Exception as exp_err:
                warnings.append(f"Fehler beim Verfall {exp}: {exp_err}")
                continue

        # Ausgabe von Warnungen (falls vorhanden)
        if warnings:
            st.warning("Einige Zeilen wurden übersprungen oder es gab Probleme. Beispiele:")
            for w in warnings[:8]:
                st.write("- " + str(w))
            if len(warnings) > 8:
                st.write(f"... und {len(warnings)-8} weitere Warnungen")

        # DataFrame & Filter
        df = pd.DataFrame(rows)
        if df.empty:
            st.warning("Keine passenden Optionen gefunden (nach Filter/Validierung).")
            st.stop()

        if not show_all:
            df = df[(df["Laufzeit (Tage)"] >= 25) & (df["Laufzeit (Tage)"] <= 60)]

        if df.empty:
            st.warning("Keine passenden Optionen mit 25–60 Tagen Laufzeit gefunden (nach Filter).")
            st.stop()

        # Sort & round
        df = df.sort_values(by=["Laufzeit (Tage)", "Strike"])
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].round(2)

        # Insert blank rows between different Laufzeiten
        separated_rows = []
        last_days = None
        for _, row in df.iterrows():
            if last_days is not None and row["Laufzeit (Tage)"] != last_days:
                separated_rows.append({col: "" for col in df.columns})
            separated_rows.append(row)
            last_days = row["Laufzeit (Tage)"]
        df_display = pd.DataFrame(separated_rows)

        # Highlight Rendite p.a.
        def highlight_roi(row):
            style = [''] * len(row)
            if "Rendite p.a. (%)" in df_display.columns:
                style[df_display.columns.get_loc("Rendite p.a. (%)")] = "font-weight: bold; color: #006400"
            return style

        st.subheader(f"📋 Ergebnisse für {ticker} (Kurs: {current_price:.2f} $)")
        st.dataframe(df_display.style.apply(highlight_roi, axis=1).format(precision=2), use_container_width=True)

        # Chart
        plt.figure(figsize=(10, 6))
        for strike in strikes:
            subset = df[df["Strike"] == strike]
            if not subset.empty:
                plt.plot(subset["Laufzeit (Tage)"], subset["Rendite p.a. (%)"], marker="o", label=f"Strike {strike}$")
        plt.title(f"Put-Rendite p.a. für {ticker}")
        plt.xlabel("Laufzeit (Tage)")
        plt.ylabel("Rendite p.a. (%)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Unerwarteter Fehler: {e}")
