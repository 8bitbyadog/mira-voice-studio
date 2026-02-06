#!/usr/bin/env python3
"""
Mira Voice Studio - Setup

A local TTS app that generates voiceovers with synced SRT captions.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if it exists
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="mira-voice-studio",
    version="0.1.0",
    author="Mira Voice Studio",
    description="Local TTS app that generates voiceovers with synced SRT captions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mira-voice-studio/mira-voice-studio",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "TTS>=0.22.0",
        "openai-whisper>=20231117",
        "pydub>=0.25.1",
        "soundfile>=0.12.1",
        "nltk>=3.8.1",
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "gradio>=4.0.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "voice_studio=voice_studio.cli:main",
            "voice_studio_ui=voice_studio.ui.run:main",
            "mira=voice_studio.cli:main",
            "mira_ui=voice_studio.ui.run:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Video",
    ],
    keywords="tts text-to-speech voice audio captions srt subtitles",
)
