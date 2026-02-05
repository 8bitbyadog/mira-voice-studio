#!/usr/bin/env python3
"""
Model downloader for Mira Voice Studio.

Downloads pretrained models for:
- GPT-SoVITS (voice cloning)
- Whisper (alignment)
- Coqui TTS (fallback)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import urllib.request
import hashlib
import json

# Model URLs and checksums
MODELS = {
    "gptsovits": {
        "base": {
            "description": "GPT-SoVITS base pretrained model",
            "files": {
                "gpt_weights.ckpt": {
                    "url": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                    "size_mb": 500,
                },
                "sovits_weights.pth": {
                    "url": "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2G2333k.pth",
                    "size_mb": 300,
                },
            }
        }
    },
    "whisper": {
        "tiny": {
            "description": "Whisper tiny model (~75MB)",
            "auto_download": True,  # Whisper downloads automatically
        },
        "base": {
            "description": "Whisper base model (~140MB)",
            "auto_download": True,
        },
        "small": {
            "description": "Whisper small model (~460MB)",
            "auto_download": True,
        },
    },
    "coqui": {
        "default": {
            "description": "Coqui TTS default English model",
            "auto_download": True,  # Coqui downloads automatically
        }
    }
}


def get_models_dir() -> Path:
    """Get the models directory."""
    return Path.home() / "mira_voice_studio" / "models"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """
    Download a file with progress display.

    Args:
        url: URL to download from.
        dest: Destination path.
        desc: Description for progress display.

    Returns:
        True if successful.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {desc or dest.name}...")
        print(f"  URL: {url}")
        print(f"  Destination: {dest}")

        # Download with progress
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

        urllib.request.urlretrieve(url, dest, reporthook=report_progress)
        print()  # New line after progress

        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_gptsovits_models(models_dir: Path) -> bool:
    """Download GPT-SoVITS pretrained models."""
    print("\n" + "=" * 50)
    print("Downloading GPT-SoVITS pretrained models")
    print("=" * 50)

    pretrained_dir = models_dir / "pretrained" / "gptsovits_base"
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    model_info = MODELS["gptsovits"]["base"]
    success = True

    for filename, info in model_info["files"].items():
        dest = pretrained_dir / filename

        if dest.exists():
            print(f"\n{filename} already exists, skipping...")
            continue

        print(f"\nDownloading {filename} (~{info['size_mb']}MB)...")
        if not download_file(info["url"], dest, filename):
            success = False

    # Create config file
    config = {
        "name": "gptsovits_base",
        "description": "GPT-SoVITS v2 pretrained model",
        "type": "pretrained",
        "language": "multilingual",
    }
    config_path = pretrained_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if success:
        print(f"\nGPT-SoVITS models downloaded to: {pretrained_dir}")
    return success


def setup_whisper(model_size: str = "base") -> bool:
    """
    Ensure Whisper model is downloaded.

    Whisper models are downloaded automatically on first use,
    but we can trigger the download here.
    """
    print(f"\nChecking Whisper {model_size} model...")

    try:
        import whisper
        print(f"  Downloading/verifying Whisper {model_size}...")
        whisper.load_model(model_size)
        print(f"  Whisper {model_size} ready!")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def setup_coqui() -> bool:
    """
    Ensure Coqui TTS default model is downloaded.

    Coqui models are downloaded automatically on first use.
    """
    print("\nChecking Coqui TTS default model...")

    try:
        from TTS.api import TTS
        print("  Downloading/verifying Coqui TTS model...")
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)
        del tts
        print("  Coqui TTS ready!")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained models for Mira Voice Studio"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["all", "gptsovits", "whisper", "coqui"],
        default="all",
        help="Type of models to download"
    )
    parser.add_argument(
        "--whisper-size",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size to download"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Custom models directory"
    )

    args = parser.parse_args()

    models_dir = args.models_dir or get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Mira Voice Studio - Model Downloader")
    print(f"Models directory: {models_dir}")

    results = {}

    if args.type in ["all", "coqui"]:
        results["coqui"] = setup_coqui()

    if args.type in ["all", "whisper"]:
        results["whisper"] = setup_whisper(args.whisper_size)

    if args.type in ["all", "gptsovits"]:
        results["gptsovits"] = download_gptsovits_models(models_dir)

    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)

    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    if all(results.values()):
        print("\nAll models ready!")
        return 0
    else:
        print("\nSome downloads failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
