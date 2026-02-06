#!/usr/bin/env python3
"""
Launch script for Mira Voice Studio UI.

Usage:
    python -m voice_studio.ui.run
    # or
    voice_studio_ui
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch Mira Voice Studio Web UI"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Port to run on (default: 7860)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Mira Voice Studio")
    print("=" * 50)
    print(f"Starting server on http://{args.host}:{args.port}")
    if args.share:
        print("Creating public link...")
    print()

    from voice_studio.ui import launch

    launch(
        server_port=args.port,
        server_name=args.host,
        share=args.share,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
