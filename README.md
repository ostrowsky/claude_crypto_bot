# Claude Crypto Bot

## How It Works

### 1. Market Scan
The bot scans the configured watchlist and evaluates each symbol using the strategy engine.

### 2. Signal Filtering
Signals are filtered by:
- trend
- volatility
- RSI
- ADX
- range extension
- impulse conditions

### 3. Telegram Delivery
The bot presents:
- current monitoring status  
- number of watchlist coins  
- hot coins confirmed today  
- active positions  

### 4. Monitoring Loop
Once started, the monitoring process continuously checks active positions and updates state.

### 5. Dataset Logging
A separate collector logs the latest closed bar for each tracked symbol and later fills forward labels for ML evaluation.

### 6. ML Acceptance Gate
Experimental ML ranking is not automatically enabled in production mode.  
A separate worker checks validation quality and enables runtime ranking only if acceptance thresholds are met.

---

## Installation

### 1. Clone the repository
git clone https://github.com/ostrowsky/claude_crypto_bot.git
cd claude_crypto_bot

### 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

Windows:
.venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

---

## Configuration

Create a .env file and add your Telegram bot token:

TELEGRAM_BOT_TOKEN=your_token_here

Main runtime settings are stored in config.py, including:
- Binance REST endpoint  
- watchlist  
- timeframes  
- scan frequency  
- indicator periods  
- entry thresholds  
- impulse scanner settings  
- exit logic  
- ML ranker toggles  
- acceptance-gate thresholds  

---

## Running the Bot

python bot.py

---

## Typical Workflow

1. Start the Telegram bot  
2. Run market scan  
3. Review active setups  
4. Start monitoring  
5. Track open positions in Telegram  
6. Optionally run data collection and ML evaluation scripts separately  

---

## Dataset / ML Workflow

The project includes an experimental ML pipeline for ranking candidates.

### Data collection
Bar snapshots are collected on a fixed cycle and saved with forward labels.

### Candidate ranking
The ranking layer can score candidates using learned signals.

### Safe rollout
Runtime ranking is protected by an acceptance gate that checks:
- validation AUC  
- top-1 delta  
- minimum amount of live data  

---

## Notes

- This project is designed for research and experimentation  
- It is not financial advice  
- Exchange availability, symbol set, and API limits may change over time  
- Before using in production, validate thresholds and add risk controls  

---

## Future Improvements

- cleaner config separation (.env + yaml/json)  
- better tests for strategy components  
- Docker support  
- richer backtest reports  
- model versioning for ML ranker  
- portfolio/risk module  
- exchange abstraction beyond Binance  

---

## Disclaimer

This software is provided for educational and research purposes only. Use it at your own risk.
