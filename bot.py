# bot.py
import ccxt
import numpy as np
import pandas as pd
import time
from datetime import datetime
import ta
import threading
import os
import pytz

# Configuration
SYMBOLS = [
    "TIA/USDT", "ATOM/USDT", "SOL/USDT", "ENA/USDT", "POPCAT/USDT", "BTC/USDT", "AAVE/USDT", "LINK/USDT",
    "NEAR/USDT", "SUI/USDT", "PEPE/USDT", "SHIB/USDT", "ETH/USDT", "XRP/USDT", "TAO/USDT", "SEI/USDT",
    "INJ/USDT", "FET/USDT", "DOGE/USDT", "ADA/USDT", "CRV/USDT", "PYTH/USDT", "BNB/USDT",
    "GALA/USDT", "ONDO/USDT"
]
MAX_POSITIONS = 5
SYMBOL_PARAMS = {
    "default": {"risk_per_trade": 0.01, "leverage": 10, "tp_ratio": 2.0, "sl_ratio": 1.0},
}
for symbol in SYMBOLS:
    SYMBOL_PARAMS[symbol] = SYMBOL_PARAMS["default"]
MIN_ATR = {
    "default": 0.0001, "TIA/USDT": 0.01, "ATOM/USDT": 0.01, "SOL/USDT": 0.1, "ENA/USDT": 0.001,
    "POPCAT/USDT": 0.001, "BTC/USDT": 50.0, "AAVE/USDT": 0.1, "LINK/USDT": 0.01, "NEAR/USDT": 0.01,
    "SUI/USDT": 0.01, "PEPE/USDT": 0.00000001, "SHIB/USDT": 0.00000001, "ETH/USDT": 10.0,
    "XRP/USDT": 0.001, "TAO/USDT": 0.1, "SEI/USDT": 0.001, "INJ/USDT": 0.01, "FET/USDT": 0.01,
    "DOGE/USDT": 0.0001, "ADA/USDT": 0.001, "CRV/USDT": 0.001, "PYTH/USDT": 0.001,
    "BNB/USDT": 1.0, "GALA/USDT": 0.0001, "ONDO/USDT": 0.001,
}

# Initialisation de l'exchange
exchange = ccxt.mexc({"timeout": 120000, "enableRateLimit": True})

# Fuseau horaire France (CEST)
tz_paris = pytz.timezone('Europe/Paris')

# Gestion des fichiers
positions_file = "positions.csv"
trades_file = "trades.csv"
stats_file = "stats.csv"
missed_trades_file = "missed_trades.txt"
positions_backup_file = "positions_backup.csv"
trades_backup_file = "trades_backup.csv"
stats_backup_file = "stats_backup.csv"

positions_columns = [
    "Symbole", "Type", "Prix_Entree", "Quantite", "TP", "SL", "RSI", "EMA_30", "ATR", "ADX",
    "Fib_1618", "Temps_Entree", "Position_ID", "Marge", "Levier"
]
trades_columns = [
    "Symbole", "Type", "Prix_Entree", "Prix_Sortie", "Quantite", "PNL", "Raison_Sortie",
    "RSI_Sortie", "EMA_30_Sortie", "ATR_Sortie", "Temps_Entree", "Temps_Sortie", "Position_ID",
    "Marge", "Levier"
]
stats_columns = [
    "Total_Trades", "Wins", "Losses", "Winrate", "Total_PNL", "Max_Drawdown", "Sharpe_Ratio",
    "Update_Time"
]

# Initialisation des fichiers CSV
for file, columns in [
    (positions_file, positions_columns),
    (trades_file, trades_columns),
    (stats_file, stats_columns),
    (positions_backup_file, positions_columns),
    (trades_backup_file, trades_columns),
    (stats_backup_file, stats_columns),
]:
    if not os.path.exists(file):
        pd.DataFrame(columns=columns).to_csv(file, index=False)

# Initialisation missed_trades.txt
if not os.path.exists(missed_trades_file):
    with open(missed_trades_file, "w") as f:
        f.write("Trades manqués par raison:\n")

# Variables globales
positions = []
positions_lock = threading.Lock()
trades = []
stats = {
    "Total_Trades": 0,
    "Wins": 0,
    "Losses": 0,
    "Winrate": 0.0,
    "Total_PNL": 0.0,
    "Max_Drawdown": 0.0,
    "Sharpe_Ratio": 0.0,
}
missed_trades_reasons = {}

# Fonctions utilitaires
def calculate_atr(highs, lows, closes, period=14):
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]), np.abs(closes[:-1] - lows[1:]))
    atr = np.mean(tr[-period:])
    return atr

def detect_market_structure(ohlcv):
    highs = ohlcv[:, 2]
    lows = ohlcv[:, 3]
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return "Bullish"
    elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return "Bearish"
    return "Neutral"

def detect_breaker_block(ohlcv):
    highs = ohlcv[:, 2]
    lows = ohlcv[:, 3]
    closes = ohlcv[:, 4]
    for i in range(-3, -1):
        if closes[i] < lows[i-1] and closes[i-1] > highs[i-2]:
            return lows[i-1], "Bullish"
        if closes[i] > highs[i-1] and closes[i-1] < lows[i-2]:
            return highs[i-1], "Bearish"
    return None, None

def calculate_fibonacci(price, high, low):
    diff = high - low
    fib_0_5 = high - 0.5 * diff
    fib_0_618 = high - 0.618 * diff
    fib_0_705 = high - 0.705 * diff
    fib_0_79 = high - 0.79 * diff
    fib_0_9 = high - 0.9 * diff
    fib_1618 = high + 0.618 * diff
    return fib_0_5, fib_0_618, fib_0_705, fib_0_79, fib_0_9, fib_1618

def save_positions():
    with positions_lock:
        pd.DataFrame(positions).to_csv(positions_file, index=False)
        pd.DataFrame(positions).to_csv(positions_backup_file, index=False)
        print(f"[{datetime.now(tz_paris)}] Écriture réussie dans positions.csv")
        with open("console_log.txt", "a") as f:
            f.write(f"[{datetime.now(tz_paris)}] Écriture réussie dans positions.csv\n")

def save_trades():
    pd.DataFrame(trades).to_csv(trades_file, index=False)
    pd.DataFrame(trades).to_csv(trades_backup_file, index=False)
    print(f"[{datetime.now(tz_paris)}] Écriture réussie dans trades.csv")
    with open("console_log.txt", "a") as f:
        f.write(f"[{datetime.now(tz_paris)}] Écriture réussie dans trades.csv\n")

def save_stats():
    stats["Update_Time"] = str(datetime.now(tz_paris))
    stats_df = pd.DataFrame([stats], columns=stats_columns)
    stats_df.to_csv(stats_file, index=False)
    stats_df.to_csv(stats_backup_file, index=False)
    with open(missed_trades_file, "w") as f:
        f.write("Trades manqués par raison:\n")
        for reason, count in missed_trades_reasons.items():
            f.write(f"- {reason}: {count}\n")
    print(f"[{datetime.now(tz_paris)}] Écriture réussie dans stats.csv et missed_trades.txt")
    with open("console_log.txt", "a") as f:
        f.write(f"[{datetime.now(tz_paris)}] Écriture réussie dans stats.csv et missed_trades.txt\n")

def keep_alive():
    while True:
        print(f"[{datetime.now(tz_paris)}] Keep alive...")
        with open("console_log.txt", "a") as f:
            f.write(f"[{datetime.now(tz_paris)}] Keep alive...\n")
        time.sleep(60)

def main():
    print(f"[{datetime.now(tz_paris)}] Tous les imports réussis")
    with open("console_log.txt", "a") as f:
        f.write(f"[{datetime.now(tz_paris)}] Tous les imports réussis\n")
    threading.Thread(target=keep_alive, daemon=True).start()

    while True:
        print(f"[{datetime.now(tz_paris)}] Début de la boucle principale")
        with open("console_log.txt", "a") as f:
            f.write(f"[{datetime.now(tz_paris)}] Début de la boucle principale\n")

        for symbol in SYMBOLS:
            print(f"[{datetime.now(tz_paris)}] Analyse de {symbol}")
            with open("console_log.txt", "a") as f:
                f.write(f"[{datetime.now(tz_paris)}] Analyse de {symbol}\n")
            try:
                with positions_lock:
                    if len(positions) >= MAX_POSITIONS:
                        print(f"[{datetime.now(tz_paris)}] {symbol} : Max positions atteint ({MAX_POSITIONS})")
                        with open("console_log.txt", "a") as f:
                            f.write(f"[{datetime.now(tz_paris)}] {symbol} : Max positions atteint ({MAX_POSITIONS})\n")
                        continue
                    if sum(1 for p in positions if p["Symbole"] == symbol) > 0:
                        print(f"[{datetime.now(tz_paris)}] {symbol} : Doublon détecté, skip")
                        with open("console_log.txt", "a") as f:
                            f.write(f"[{datetime.now(tz_paris)}] {symbol} : Doublon détecté, skip\n")
                        continue

                params = SYMBOL_PARAMS.get(symbol, SYMBOL_PARAMS["default"])
                min_atr = MIN_ATR.get(symbol, MIN_ATR["default"])

                ohlcv_1h = exchange.fetch_ohlcv(symbol, "1h", limit=200)
                ohlcv_15m = exchange.fetch_ohlcv(symbol, "15m", limit=200)
                if len(ohlcv_1h) < 50 or len(ohlcv_15m) < 50:
                    print(f"[{datetime.now(tz_paris)}] {symbol} : Données insuffisantes")
                    with open("console_log.txt", "a") as f:
                        f.write(f"[{datetime.now(tz_paris)}] {symbol} : Données insuffisantes\n")
                    continue

                ohlcv_1h = np.array(ohlcv_1h)
                ohlcv_15m = np.array(ohlcv_15m)
                closes_1h, highs_1h, lows_1h = ohlcv_1h[:, 4], ohlcv_1h[:, 2], ohlcv_1h[:, 3]
                closes_15m, highs_15m, lows_15m, volumes_15m = ohlcv_15m[:, 4], ohlcv_15m[:, 2], ohlcv_15m[:, 3], ohlcv_15m[:, 5]

                price = round(closes_15m[-1], 8)
                atr_15m = calculate_atr(highs_15m, lows_15m, closes_15m)
                mean_volume = np.mean(volumes_15m[-20:])

                market_structure = detect_market_structure(ohlcv_1h)
                bb_price, bb_type = detect_breaker_block(ohlcv_15m)
                fib_0_5, fib_0_618, fib_0_705, fib_0_79, fib_0_9, fib_1618 = calculate_fibonacci(price, max(highs_1h[-50:]), min(lows_1h[-50:]))
                rsi = ta.momentum.RSIIndicator(pd.Series(closes_15m)).rsi().iloc[-1]
                ema_30 = ta.trend.EMAIndicator(pd.Series(closes_15m), window=30).ema_indicator().iloc[-1]
                adx = ta.trend.ADXIndicator(pd.Series(highs_15m), pd.Series(lows_15m), pd.Series(closes_15m)).adx().iloc[-1]

                reasons = []
                long_condition = True
                if market_structure != "Bullish":
                    reasons.append("Market Structure pas Bullish")
                    long_condition = False
                if bb_price is None or bb_type != "Bullish":
                    reasons.append("Pas de Breaker Block Bullish")
                    long_condition = False
                if bb_price is not None and abs(price - bb_price) / price >= 0.07:
                    reasons.append("Prix trop éloigné du Breaker Block")
                    long_condition = False
                if not (fib_0_5 <= price <= fib_0_9):
                    reasons.append("Prix hors de la zone Fibonacci [0.5-0.9]")
                    long_condition = False
                if atr_15m < min_atr * 0.25:
                    reasons.append(f"ATR trop faible ({atr_15m:.8f} < {min_atr * 0.25:.8f})")
                    long_condition = False
                if volumes_15m[-1] <= mean_volume * 0.5:
                    reasons.append(f"Volume trop faible ({volumes_15m[-1]:.2f} < {mean_volume * 0.5:.2f})")
                    long_condition = False

                short_condition = True
                if market_structure != "Bearish":
                    reasons.append("Market Structure pas Bearish")
                    short_condition = False
                if bb_price is None or bb_type != "Bearish":
                    reasons.append("Pas de Breaker Block Bearish")
                    short_condition = False
                if bb_price is not None and abs(price - bb_price) / price >= 0.07:
                    reasons.append("Prix trop éloigné du Breaker Block")
                    short_condition = False
                if not (fib_0_5 <= price <= fib_0_9):
                    reasons.append("Prix hors de la zone Fibonacci [0.5-0.9]")
                    short_condition = False
                if atr_15m < min_atr * 0.25:
                    reasons.append(f"ATR trop faible ({atr_15m:.8f} < {min_atr * 0.25:.8f})")
                    short_condition = False
                if volumes_15m[-1] <= mean_volume * 0.5:
                    reasons.append(f"Volume trop faible ({volumes_15m[-1]:.2f} < {mean_volume * 0.5:.2f})")
                    short_condition = False

                if not long_condition and not short_condition:
                    for reason in reasons:
                        missed_trades_reasons[reason] = missed_trades_reasons.get(reason, 0) + 1
                    print(f"[{datetime.now(tz_paris)}] {symbol} : Aucune condition d'entrée remplie")
                    print(f"  Raison(s) : {', '.join(reasons)}")
                    with open("console_log.txt", "a") as f:
                        f.write(f"[{datetime.now(tz_paris)}] {symbol} : Aucune condition d'entrée remplie\n")
                        f.write(f"  Raison(s) : {', '.join(reasons)}\n")
                    continue

                if long_condition:
                    position_type = "Long"
                    sl_price = price - atr_15m * params["sl_ratio"]
                    tp_price = price + atr_15m * params["tp_ratio"]
                elif short_condition:
                    position_type = "Short"
                    sl_price = price + atr_15m * params["sl_ratio"]
                    tp_price = price - atr_15m * params["tp_ratio"]

                margin = 100.0
                leverage = 10.0
                position_size = margin * leverage
                quantity = position_size / price

                position = {
                    "Symbole": symbol,
                    "Type": position_type,
                    "Prix_Entree": price,
                    "Quantite": quantity,
                    "TP": tp_price,
                    "SL": sl_price,
                    "RSI": rsi,
                    "EMA_30": ema_30,
                    "ATR": atr_15m,
                    "ADX": adx,
                    "Fib_1618": fib_1618,
                    "Temps_Entree": str(datetime.now(tz_paris)),
                    "Position_ID": f"{symbol}_{datetime.now().timestamp()}",
                    "Marge": margin,
                    "Levier": leverage
                }

                with positions_lock:
                    positions.append(position)
                    save_positions()

                print(f"[{datetime.now(tz_paris)}] {symbol} {position_type} entré: Price={price}, Marge={margin} USDT, Levier={leverage}x")
                with open("console_log.txt", "a") as f:
                    f.write(f"[{datetime.now(tz_paris)}] {symbol} {position_type} entré: Price={price}, Marge={margin} USDT, Levier={leverage}x\n")

                # Gestion des sorties (SL/TP)
                while True:
                    current_price = exchange.fetch_ticker(symbol)["last"]
                    with positions_lock:
                        for pos in positions:
                            if pos["Symbole"] == symbol and pos["Position_ID"] == position["Position_ID"]:
                                if pos["Type"] == "Long":
                                    if current_price >= pos["TP"]:
                                        reason = "TP Hit"
                                        break
                                    if current_price <= pos["SL"]:
                                        reason = "SL Hit"
                                        break
                                else:
                                    if current_price <= pos["TP"]:
                                        reason = "TP Hit"
                                        break
                                    if current_price >= pos["SL"]:
                                        reason = "SL Hit"
                                        break
                        else:
                            continue
                        break

                # Calcul du PNL
                pnl = (current_price - pos["Prix_Entree"]) * pos["Quantite"] if pos["Type"] == "Long" else (pos["Prix_Entree"] - current_price) * pos["Quantite"]
                trade = {
                    "Symbole": symbol,
                    "Type": pos["Type"],
                    "Prix_Entree": pos["Prix_Entree"],
                    "Prix_Sortie": current_price,
                    "Quantite": pos["Quantite"],
                    "PNL": pnl,
                    "Raison_Sortie": reason,
                    "RSI_Sortie": ta.momentum.RSIIndicator(pd.Series(closes_15m)).rsi().iloc[-1],
                    "EMA_30_Sortie": ta.trend.EMAIndicator(pd.Series(closes_15m), window=30).ema_indicator().iloc[-1],
                    "ATR_Sortie": atr_15m,
                    "Temps_Entree": pos["Temps_Entree"],
                    "Temps_Sortie": str(datetime.now(tz_paris)),
                    "Position_ID": pos["Position_ID"],
                    "Marge": margin,
                    "Levier": leverage
                }

                with positions_lock:
                    positions.remove(pos)
                    trades.append(trade)
                    save_positions()
                    save_trades()

                stats["Total_Trades"] += 1
                if pnl > 0:
                    stats["Wins"] += 1
                else:
                    stats["Losses"] += 1
                stats["Winrate"] = stats["Wins"] / stats["Total_Trades"] if stats["Total_Trades"] > 0 else 0.0
                stats["Total_PNL"] += pnl
                stats["Update_Time"] = str(datetime.now(tz_paris))
                save_stats()

                print(f"[{datetime.now(tz_paris)}] {symbol} {pos['Type']} sorti: Price={current_price}, PNL={pnl:.2f} USDT, Total_PNL={stats['Total_PNL']:.2f} USDT, Reason={reason}")
                with open("console_log.txt", "a") as f:
                    f.write(f"[{datetime.now(tz_paris)}] {symbol} {pos['Type']} sorti: Price={current_price}, PNL={pnl:.2f} USDT, Total_PNL={stats['Total_PNL']:.2f} USDT, Reason={reason}\n")

            except Exception as e:
                print(f"[{datetime.now(tz_paris)}] Erreur {symbol}: {e}")
                with open("console_log.txt", "a") as f:
                    f.write(f"[{datetime.now(tz_paris)}] Erreur {symbol}: {e}\n")
                time.sleep(5)

        time.sleep(30)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"[{datetime.now(tz_paris)}] ===== ARRÊT DU BOT =====")
        with open("console_log.txt", "a") as f:
            f.write(f"[{datetime.now(tz_paris)}] ===== ARRÊT DU BOT =====\n")
        print(f"[{datetime.now(tz_paris)}] Sauvegarde des données...")
        with open("console_log.txt", "a") as f:
            f.write(f"[{datetime.now(tz_paris)}] Sauvegarde des données...\n")
        save_positions()
        save_trades()
        save_stats()
