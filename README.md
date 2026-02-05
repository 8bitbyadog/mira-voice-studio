# Mira Voice Studio

A local TTS application that generates voiceovers with synced SRT captions. Built for content creators who want to type a script, get audio + captions, and drop them into video editors like DaVinci Resolve.

## Features

- **Text-to-Speech**: Generate natural-sounding voiceovers from text
- **Synced Captions**: Automatically create SRT/VTT files with word-level timing
- **Apple Silicon Optimized**: Uses MPS (Metal Performance Shaders) for GPU acceleration
- **Flexible Output**: Master files by default, with optional chunk and selection exports
- **I/O Selection**: Export specific portions of your audio with adjusted timestamps

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- 8GB+ RAM recommended

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mira-voice-studio.git
cd mira-voice-studio

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

## Quick Start

```bash
# Generate voiceover from a text file
voice_studio -i script.txt

# Generate with word-level captions
voice_studio -i script.txt --word-level

# List available voices
voice_studio --list-voices

# Use a specific voice and speed
voice_studio -i script.txt --voice vits --speed 1.1
```

## Output Structure

By default, generation produces only the master file:

```
output/2024-02-05_my_script/
├── master/
│   ├── my_script.wav           # Full stitched audio
│   ├── my_script.srt           # Sentence-level captions
│   ├── my_script_words.srt     # Word-level captions
│   └── my_script.vtt           # Web format
└── manifest.json               # Metadata
```

### Export Options

```bash
# Export all individual sentence chunks
voice_studio -i script.txt --export-chunks

# Export a specific time selection
voice_studio -i script.txt --in-time 0:32 --out-time 1:45
```

## CLI Reference

```
usage: voice_studio [-h] --input INPUT [--output OUTPUT] [--voice VOICE]
                    [--speed SPEED] [--pause PAUSE] [--word-level]
                    [--export-chunks] [--in-time IN] [--out-time OUT]
                    [--list-voices]

Options:
  --input, -i       Input text file or quoted string (required)
  --output, -o      Output directory (default: ~/Videos/VO/)
  --voice, -v       Voice model name (default: default)
  --speed, -s       Speaking speed 0.5-2.0 (default: 1.0)
  --pause, -p       Pause between sentences in ms (default: 300)
  --word-level, -w  Include word-level captions
  --export-chunks   Export individual sentence chunks
  --in-time         Selection start time (e.g., "0:32" or "32")
  --out-time        Selection end time (e.g., "1:45" or "105")
  --list-voices     List available voice models
  --whisper-model   Whisper model size (tiny/base/small/medium/large)
```

## Development Roadmap

- [x] **Phase 1**: Core pipeline (CLI) with Coqui TTS
- [ ] **Phase 2**: GPT-SoVITS integration
- [ ] **Phase 3**: Gradio UI - Generate tab
- [ ] **Phase 4**: Gradio UI - Train tab (recording, import)
- [ ] **Phase 5**: Gradio UI - Models tab
- [ ] **Phase 6**: Gradio UI - Settings tab
- [ ] **Phase 7**: Training backend
- [ ] **Phase 8**: Automation lanes

## License

MIT License
