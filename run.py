import os
import sys
import webbrowser
import threading
import time
import argparse
import uvicorn


def open_browser_after_delay(url: str, delay: float = 1.5):
    def _open():
        time.sleep(delay)
        print(f"  Opening browser at: {url}")
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def print_banner():
    print()
    print("=" * 70)
    print("       AlphaTrade AI - Stock Prediction & Trading Terminal")
    print("          7-Day Multi-Horizon Forecasting Engine")
    print("=" * 70)
    print()
    print("  Models:  XGBoost (7-day)  |  LSTM Neural  |  GRU Recurrent")
    print("  Engine:  252-Day Rolling Normalization  |  19 NSE Equities")
    print("  Valid.:  1,026-Window Walk-Forward  |  70.67% Day-1 Direction")
    print()


def print_endpoints():
    print("  +----------------------------------------------------------+")
    print("  |  REST API Endpoints                                      |")
    print("  +----------------------------------------------------------+")
    print("  |  GET  /api/health              System health check       |")
    print("  |  GET  /api/system/status       Full system status        |")
    print("  |  GET  /api/stocks              Curated watchlist         |")
    print("  |  GET  /api/stock/{t}/overview  Stock fundamentals        |")
    print("  |  GET  /api/stock/{t}/history   OHLCV + indicators        |")
    print("  |  GET  /api/stock/{t}/predict   7-day AI forecast         |")
    print("  |  GET  /api/benchmark           Model benchmarks          |")
    print("  |  GET  /api/ipo/upcoming        IPO analysis list         |")
    print("  |  GET  /api/ipo/history         IPO performance           |")
    print("  |  GET  /api/ipo/{name}          Detailed IPO report       |")
    print("  |  GET  /api/sentiment/{t}       Stock sentiment           |")
    print("  |  GET  /api/sentiment/market/mood Market mood index       |")
    print("  |  GET  /api/reports/insights/{t} Stock research           |")
    print("  |  POST /api/reports/analyze     Document analysis         |")
    print("  +----------------------------------------------------------+")
    print()


def main():
    parser = argparse.ArgumentParser(description="AlphaTrade AI Server")
    parser.add_argument("--train", action="store_true",
                        help="Run model training before starting the server")
    parser.add_argument("--demo", action="store_true",
                        help="Force demo mode (no live data fetching)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser on startup")
    args = parser.parse_args()

    print_banner()

    # Set demo mode environment variable if requested
    if args.demo:
        os.environ["ALPHATRADE_DEMO_MODE"] = "1"
        print("  ⚠  DEMO MODE ENABLED — All data will be served from fixtures")
        print()

    # Run training if requested
    if args.train:
        print("  Running model training pipeline...")
        print("-" * 70)
        try:
            from backend.train_models import train_and_save_all
            train_and_save_all()
        except Exception as e:
            print(f"  Training failed: {e}")
            print("  Continuing with existing model weights...")
        print("-" * 70)
        print()

    print_endpoints()

    url = f"http://127.0.0.1:{args.port}"
    print(f"  Server URL:  {url}")
    print(f"  Press Ctrl+C to stop the server.")
    print("-" * 70)

    # Schedule auto-open in default browser
    if not args.no_browser:
        open_browser_after_delay(url, delay=1.5)

    # Start Uvicorn Server
    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=args.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
