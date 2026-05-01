"""
Client starter script for AirHub federated learning.

Usage:
    python start_client.py --city <city> --country <country> [--persistent]

Options:
    --city          City name (default: from config.DEFAULT_CITY)
    --country       Country code (default: from config.DEFAULT_COUNTRY)
    --persistent    Keep client running and auto-reconnect (for continuous training)
    --server        Custom server address (default: from config.FL_SERVER_ADDRESS)

Examples:
    python start_client.py --city London --country GB
    python start_client.py --city London --country GB --persistent
    python start_client.py --city London --country GB --server 192.168.1.100:8080
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.client_node import start_client
from utils.logger import setup_logger
import config

logger = setup_logger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start AirHub federated learning client")
    parser.add_argument("--city", type=str, default=config.DEFAULT_CITY,
                        help=f"City name (default: {config.DEFAULT_CITY})")
    parser.add_argument("--country", type=str, default=config.DEFAULT_COUNTRY,
                        help=f"Country code (default: {config.DEFAULT_COUNTRY})")
    parser.add_argument("--server", type=str, default=config.FL_SERVER_ADDRESS,
                        help=f"Flower server address (default: {config.FL_SERVER_ADDRESS})")
    parser.add_argument("--persistent", action="store_true",
                        help="Keep client running and auto-reconnect")

    args = parser.parse_args()

    if args.persistent:
        import time
        logger.info(f"Starting PERSISTENT client for {args.city}, {args.country}")
        logger.info(f"Connecting to server: {args.server}")
        logger.info("Press Ctrl+C to stop")

        while True:
            try:
                start_client(args.city, args.country, server_address=args.server)
                logger.info("Client connection ended, reconnecting in 10 seconds...")
                time.sleep(10)
            except KeyboardInterrupt:
                logger.info("Persistent client shutting down")
                break
            except Exception as e:
                logger.error(f"Client error: {e}, reconnecting in 10 seconds...")
                time.sleep(10)
    else:
        logger.info(f"Starting client for {args.city}, {args.country}")
        logger.info(f"Connecting to server: {args.server}")
        start_client(args.city, args.country, server_address=args.server)


if __name__ == "__main__":
    main()
