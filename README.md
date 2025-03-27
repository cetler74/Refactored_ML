# ML-Powered Crypto Trading Bot

A sophisticated trading bot leveraging machine learning for cryptocurrency trading on various exchanges.

## Core Features
- Support for both simulation and live trading
- Focus on USDC trading pairs
- Machine learning models for market prediction
- Real-time monitoring dashboard

## Setup
1. Create a virtual environment: `python3 -m venv trading_bot_env`
2. Activate the virtual environment: `source trading_bot_env/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `.env` file based on the `.env.example` template
5. Run the bot: `python app/main.py`

## Architecture
The bot consists of several modules:
- Bot Orchestrator: Central coordinator managing data flow and task scheduling
- ML Module: Prediction models for entry/exit signals
- Exchange Module: Handles exchange interactions
- Balance Manager: Tracks and manages balances
- Trading Pair Module: Selects optimal trading pairs
- Technical Analysis Module: Calculates indicators
- Trading Strategy Module: Defines entry/exit rules
- Configuration Module: Manages settings and sensitive data
- Database Handler: Stores and retrieves trading data
- Frontend: Provides monitoring interface

## Development
- Use `black` for code formatting
- Run tests with `pytest`
- Follow the project's coding standards

## License
MIT 