"""
Hosting script for AirHub - runs both Flower server and FastAPI server concurrently.

Usage:
    python start_hosting.py                    # Run with defaults
    python start_hosting.py --clients 3        # Run with 3 clients
    python start_hosting.py --rounds 20        # Run 20 rounds
    python start_hosting.py --non-persistent   # Stop after rounds complete

For client connections:
    - API predictions: POST http://SERVER_IP:8005/api/predict
    - Flower clients: python client_node.py --city Delhi --country IN
    - CLI predictions: python client_node.py --predict --city Delhi --country IN --api-host http://SERVER_IP:8005
"""
import sys
import os
import threading
import time
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
import config

logger = setup_logger(__name__)


def run_api_server():
    """Run FastAPI server in a thread."""
    import uvicorn
    from main import app

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        workers=1,
        reload=False,
        log_level="info"
    )


def main():
    parser = argparse.ArgumentParser(description="Start AirHub hosting server")
    parser.add_argument("--persistent", action="store_true", default=True,
                        help="Keep server running after initial rounds (default: True)")
    parser.add_argument("--clients", type=int, default=config.FL_MIN_CLIENTS,
                        help=f"Number of clients (default: {config.FL_MIN_CLIENTS})")
    parser.add_argument("--rounds", type=int, default=config.FL_ROUNDS,
                        help=f"Number of federated rounds (default: {config.FL_ROUNDS})")
    parser.add_argument("--non-persistent", action="store_true",
                        help="Disable persistent mode (server stops after rounds complete)")

    args = parser.parse_args()

    persistent = args.persistent and not args.non_persistent

    logger.info("=" * 60)
    logger.info("AirHub Hosting Server Starting...")
    logger.info("=" * 60)
    logger.info(f"Flower Server: {config.FL_SERVER_ADDRESS}")
    logger.info(f"API Server: http://{config.API_HOST}:{config.API_PORT}")
    logger.info(f"Persistent Mode: {persistent}")
    logger.info(f"Initial Clients: {args.clients}")
    logger.info(f"Initial Rounds: {args.rounds}")
    logger.info("=" * 60)

    # Start API server in a separate thread
    api_thread = threading.Thread(
        target=run_api_server,
        daemon=True
    )
    api_thread.start()

    # Wait a moment for API server to initialize
    logger.info("Waiting for API server to initialize...")
    time.sleep(2)

    # Start Flower server in main thread (blocking for signal handling)
    logger.info("Starting Flower server...")
    from federated.server_node import start_server
    
    try:
        start_server(
            min_clients=args.clients,
            max_clients=max(args.clients, 3),
            rounds=args.rounds,
            server_address=config.FL_SERVER_ADDRESS,
            persistent=persistent
        )
    except KeyboardInterrupt:
        logger.info("Shutting down AirHub hosting server...")
    except Exception as e:
        logger.error(f"Hosting server encountered an error: {e}")


if __name__ == "__main__":
    main()
