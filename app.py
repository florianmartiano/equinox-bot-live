# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import ccxt

# Configuration
st.set_page_config(page_title="Equinox Bot Cockpit", layout="wide")

# Chemins des fichiers
POSITIONS_FILE = "positions.csv"
TRADES_FILE = "trades.csv"
STATS_FILE = "stats.csv"
LOG_FILE = "console_log.txt"
MISSED_TRADES_FILE = "missed_trades.txt"

# Liste des cryptos
SYMBOLS = [
    "TIA/USDT", "ATOM/USDT", "SOL/USDT", "ENA/USDT", "POPCAT/USDT", "BTC/USDT", "AAVE/USDT", "LINK/USDT",
    "NEAR/USDT", "SUI/USDT", "PEPE/USDT", "SHIB/USDT", "ETH/USDT", "XRP/USDT", "TAO/USDT", "SEI/USDT",
    "INJ/USDT", "FET/USDT", "DOGE/USDT", "ADA/USDT", "CRV/USDT", "PYTH/USDT", "BNB/USDT",
    "GALA/USDT", "ONDO/USDT"
]

# Supprimer HYPE/USDT (non disponible sur MEXC)
if "HYPE/USDT" in SYMBOLS:
    SYMBOLS.remove("HYPE/USDT")

# Initialisation de l'exchange MEXC
exchange = ccxt.mexc({"timeout": 120000, "enableRateLimit": True})

# Fonction pour lire stats.csv
def read_stats():
    if os.path.exists(STATS_FILE):
        stats_df = pd.read_csv(STATS_FILE)
        required_columns = ["Total_Trades", "Wins", "Losses", "Winrate", "Total_PNL", "Max_Drawdown", "Sharpe_Ratio", "Update_Time"]
        if all(col in stats_df.columns for col in required_columns):
            return stats_df
        else:
            st.error("Colonnes manquantes dans stats.csv")
            return pd.DataFrame()
    return pd.DataFrame()

# Fonction pour lire missed_trades.txt
def read_missed_trades():
    missed_trades = {}
    if os.path.exists(MISSED_TRADES_FILE):
        with open(MISSED_TRADES_FILE, "r") as f:
            lines = f.readlines()
            start_reading = False
            for line in lines:
                if line.strip() == "Trades manqués par raison:":
                    start_reading = True
                    continue
                if start_reading and line.strip().startswith("- "):
                    reason = line.strip()[2:].split(": ")
                    missed_trades[reason[0]] = int(reason[1])
    return missed_trades

# Fonction pour récupérer les prix en temps réel
def fetch_prices():
    prices = {}
    for symbol in SYMBOLS:
        try:
            ticker = exchange.fetch_ticker(symbol)
            prices[symbol] = ticker["last"]
        except Exception:
            prices[symbol] = "Erreur"
    return prices

# Boucle principale pour mise à jour
def main():
    st.title("⚡ Equinox Bot Cockpit ⚡")

    # Placeholder pour le tableau des prix
    price_placeholder = st.empty()

    # Placeholder pour le reste du cockpit
    main_placeholder = st.empty()

    while True:
        # Mise à jour des prix
        with price_placeholder.container():
            st.header("Prix des cryptos en temps réel")
            prices = fetch_prices()
            price_df = pd.DataFrame({
                "Symbole": SYMBOLS,
                "Prix (USDT)": [prices.get(symbol, "Erreur") for symbol in SYMBOLS]
            })
            price_df["Couleur"] = price_df["Prix (USDT)"].apply(lambda x: "green" if isinstance(x, float) else "red")
            st.dataframe(
                price_df.style.apply(lambda x: [f"color: {x['Couleur']}"] * len(x), axis=1),
                use_container_width=True,
                hide_index=True
            )

        # Mise à jour du reste du cockpit
        with main_placeholder.container():
            # Trades en cours
            st.header("Trades en cours")
            if os.path.exists(POSITIONS_FILE):
                positions_df = pd.read_csv(POSITIONS_FILE)
                if not positions_df.empty:
                    display_columns = ["Symbole", "Type", "Prix_Entree", "TP", "SL", "Marge", "Levier", "Temps_Entree"]
                    available_columns = [col for col in display_columns if col in positions_df.columns]
                    st.dataframe(positions_df[available_columns], use_container_width=True)
                else:
                    st.write("Aucune position ouverte.")
            else:
                st.write("Fichier positions.csv introuvable.")

            # Historique des trades
            st.header("Historique des trades")
            if os.path.exists(TRADES_FILE):
                trades_df = pd.read_csv(TRADES_FILE)
                if not trades_df.empty:
                    display_columns = ["Symbole", "Type", "Prix_Entree", "Prix_Sortie", "PNL", "Raison_Sortie", "Temps_Sortie"]
                    available_columns = [col for col in display_columns if col in trades_df.columns]
                    st.dataframe(trades_df[available_columns].tail(10), use_container_width=True)
                else:
                    st.write("Aucun trade fermé.")
            else:
                st.write("Fichier trades.csv introuvable.")

            # Stats globales
            st.header("Statistiques globales")
            stats_df = read_stats()
            if not stats_df.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total PNL", f"{stats_df['Total_PNL'].iloc[-1]:.2f} USDT")
                with col2:
                    st.metric("Winrate", f"{stats_df['Winrate'].iloc[-1] * 100:.2f}%")
                with col3:
                    st.metric("Total Trades", stats_df["Total_Trades"].iloc[-1])
                st.subheader("Trades manqués par raison")
                missed_trades = read_missed_trades()
                missed_df = pd.DataFrame(missed_trades.items(), columns=["Raison", "Nombre"])
                st.dataframe(missed_df, use_container_width=True)
            else:
                st.write("Fichier stats.csv introuvable ou invalide.")

            # Performances par crypto
            st.header("Performances par crypto")
            if os.path.exists(TRADES_FILE):
                trades_df = pd.read_csv(TRADES_FILE)
                if not trades_df.empty and "PNL" in trades_df.columns and "Symbole" in trades_df.columns:
                    perf_df = trades_df.groupby("Symbole").agg({
                        "PNL": "sum",
                        "Type": "count",
                        "PNL": lambda x: (x > 0).sum()
                    }).rename(columns={"Type": "Total_Trades", "PNL": "Wins"})
                    perf_df["Winrate"] = perf_df["Wins"] / perf_df["Total_Trades"]
                    perf_df = perf_df.rename(columns={"PNL": "Total_PNL"})
                    st.dataframe(perf_df[["Total_PNL", "Total_Trades", "Wins", "Winrate"]], use_container_width=True)
                    fig = px.bar(perf_df, x=perf_df.index, y="Total_PNL", title="PNL par crypto")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Données insuffisantes ou colonne 'PNL' manquante dans trades.csv.")
            else:
                st.write("Fichier trades.csv introuvable.")

            # Logs récents
            st.header("Logs récents")
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r") as f:
                    logs = f.readlines()
                    st.text_area("Derniers logs", "".join(logs[-20:]), height=200)
            else:
                st.write("Fichier console_log.txt introuvable.")

        # Rafraîchir toutes les 5 secondes
        time.sleep(5)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
