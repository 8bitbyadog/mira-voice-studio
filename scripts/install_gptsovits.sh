#!/bin/bash
# GPT-SoVITS Installation Script for Mira Voice Studio
# Optimized for macOS with Apple Silicon

set -e

echo "========================================"
echo "GPT-SoVITS Installation for Mira Voice Studio"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${YELLOW}Warning: This script is optimized for Apple Silicon (M1/M2/M3).${NC}"
    echo "Some features may not work correctly on Intel Macs."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create models directory structure
MODELS_DIR="$HOME/mira_voice_studio/models"
mkdir -p "$MODELS_DIR/pretrained"
mkdir -p "$MODELS_DIR/custom"

echo -e "${GREEN}Created models directory at: $MODELS_DIR${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.9" ]]; then
    echo -e "${RED}Error: Python 3.9+ is required.${NC}"
    exit 1
fi

# Check for virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}Warning: No virtual environment detected.${NC}"
    echo "It's recommended to run this in a virtual environment."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Create a virtual environment with:"
        echo "  python3 -m venv venv"
        echo "  source venv/bin/activate"
        exit 1
    fi
fi

echo ""
echo "Installing GPT-SoVITS dependencies..."
echo ""

# Install PyTorch with MPS support (if not already installed)
pip install --upgrade torch torchaudio

# Install additional dependencies for GPT-SoVITS
pip install \
    transformers>=4.35.0 \
    einops>=0.7.0 \
    librosa>=0.10.0 \
    scipy>=1.11.0 \
    numba>=0.58.0 \
    g2p-en>=2.1.0 \
    pypinyin>=0.50.0 \
    cn2an>=0.5.0 \
    jieba>=0.42.0 \
    wordsegment>=1.3.1 \
    LangSegment>=0.3.0 \
    ffmpeg-python>=0.2.0 \
    PyYAML>=6.0 \
    tensorboard>=2.14.0

echo ""
echo -e "${GREEN}Dependencies installed successfully!${NC}"
echo ""

# Download pretrained models (optional)
read -p "Download pretrained GPT-SoVITS models? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading pretrained models..."
    python3 "$SCRIPT_DIR/download_models.py" --type gptsovits
    echo -e "${GREEN}Models downloaded!${NC}"
fi

echo ""
echo "========================================"
echo -e "${GREEN}GPT-SoVITS installation complete!${NC}"
echo "========================================"
echo ""
echo "Models directory: $MODELS_DIR"
echo ""
echo "Next steps:"
echo "1. Add voice models to: $MODELS_DIR/custom/"
echo "2. Each voice model needs:"
echo "   - reference.wav (3-10 seconds of speech)"
echo "   - reference.txt (transcription of the audio)"
echo "   - (optional) trained .pth model files"
echo ""
echo "Usage:"
echo "  voice_studio -i script.txt --voice my_voice"
echo ""
