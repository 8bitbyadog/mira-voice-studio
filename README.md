# Mira Voice Studio

A local TTS application that generates voiceovers with synced SRT captions. Built for content creators who want to type a script, get audio + captions, and drop them into video editors like DaVinci Resolve.

## Features

- **Text-to-Speech**: Generate natural-sounding voiceovers from text
- **Multiple TTS Engines**: Coqui TTS (simple) and GPT-SoVITS (voice cloning)
- **Voice Cloning**: Clone any voice with just a few seconds of reference audio
- **Web UI**: Beautiful Gradio interface for easy generation
- **Synced Captions**: Automatically create SRT/VTT files with word-level timing
- **Apple Silicon Optimized**: Uses MPS (Metal Performance Shaders) for GPU acceleration
- **Flexible Output**: Master files by default, with optional chunk and selection exports
- **I/O Selection**: Export specific portions of your audio with adjusted timestamps

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- 8GB+ RAM recommended (16GB+ for GPT-SoVITS)

## Installation

```bash
# Clone the repository
git clone https://github.com/8bitbyadog/mira-voice-studio.git
cd mira-voice-studio

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# (Optional) Install GPT-SoVITS for voice cloning
./scripts/install_gptsovits.sh
```

## Quick Start

### Web UI (Recommended)

```bash
# Launch the web interface
voice_studio --ui

# Or use the dedicated command
voice_studio_ui

# Custom port
voice_studio --ui --port 8080
```

Then open http://127.0.0.1:7860 in your browser.

### Command Line

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

## Web UI Features

The Gradio web interface provides:

- **Voice Selection**: Dropdown with all available voices
- **Speed Control**: Slider for speaking speed (0.5x - 2.0x)
- **Live Preview**: Audio player with waveform
- **I/O Points**: Set in/out points for partial exports
- **Export Options**:
  - Export Master (full audio + captions)
  - Export Selection (just the selected portion)
  - Export All Chunks (individual sentences)
- **Progress Tracking**: Real-time generation progress

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| H | Toggle help overlay |
| Space | Play/pause audio |
| I | Set in point |
| O | Set out point |

## TTS Engines

### Coqui TTS (Default)

Simple, reliable TTS with several built-in voices:

```bash
voice_studio -i script.txt --engine coqui --voice default
voice_studio -i script.txt --engine coqui --voice vits
```

### GPT-SoVITS (Voice Cloning)

Advanced voice cloning with reference audio:

```bash
# With a trained custom voice
voice_studio -i script.txt --engine gptsovits --voice my_voice

# Zero-shot cloning with reference audio
voice_studio -i script.txt --engine gptsovits \
  --ref-audio reference.wav \
  --ref-text "This is what I'm saying in the reference audio."
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
usage: voice_studio [-h] [--input INPUT] [--output OUTPUT] [--voice VOICE]
                    [--speed SPEED] [--pause PAUSE] [--word-level]
                    [--export-chunks] [--in-time IN] [--out-time OUT]
                    [--engine ENGINE] [--ref-audio FILE] [--ref-text TEXT]
                    [--ui] [--port PORT] [--list-voices]

Options:
  --input, -i       Input text file or quoted string
  --output, -o      Output directory (default: ~/Videos/VO/)
  --voice, -v       Voice model name (default: default)
  --speed, -s       Speaking speed 0.5-2.0 (default: 1.0)
  --pause, -p       Pause between sentences in ms (default: 300)
  --word-level, -w  Include word-level captions
  --export-chunks   Export individual sentence chunks
  --in-time         Selection start time (e.g., "0:32" or "32")
  --out-time        Selection end time (e.g., "1:45" or "105")
  --engine, -e      TTS engine: coqui, gptsovits, auto (default: auto)
  --ref-audio       Reference audio for GPT-SoVITS voice cloning
  --ref-text        Transcript of reference audio
  --ui              Launch web UI instead of CLI
  --port            Port for web UI (default: 7860)
  --list-voices     List available voice models
  --whisper-model   Whisper model size (tiny/base/small/medium/large)
```

## Voice Models

### Adding Custom GPT-SoVITS Voices

Create a folder in `~/mira_voice_studio/models/custom/` with:

```
my_voice/
├── reference.wav        # 3-10 seconds of clear speech
├── reference.txt        # Transcription of the audio
└── config.json          # Optional metadata
```

### Downloading Pretrained Models

```bash
# Download all models
python scripts/download_models.py --type all

# Download specific models
python scripts/download_models.py --type gptsovits
python scripts/download_models.py --type whisper --whisper-size base
```

## Training Your Own Voice

The Train tab provides everything you need to create custom voice models:

### Recording in the App

1. Go to **Train** → **Record**
2. Select your microphone from the dropdown
3. Click **Generate Script** to get optimized training text
4. Click **Start Session** to begin
5. Press **R** or click the button to record each line
6. Navigate with the arrows to move through the script
7. Click **Finish Session** when done

**Script Types:**
- **Mixed**: Combination of all types (recommended)
- **Phoneme**: Covers all English sounds
- **Emotional**: Various emotional expressions
- **Conversational**: YouTube/podcast style

### Importing Audio Files

1. Go to **Train** → **Import**
2. Drag and drop audio files (WAV, MP3, etc.)
3. Enable auto-transcription (uses Whisper)
4. Click **Process Files**
5. Review the created clips

### Managing Datasets

1. Go to **Train** → **Datasets**
2. Select a dataset from the dropdown
3. Review clips and toggle approval (✓/✗)
4. Configure training options
5. Click **Start Training** (Phase 7)

## Development Roadmap

- [x] **Phase 1**: Core pipeline (CLI) with Coqui TTS
- [x] **Phase 2**: GPT-SoVITS integration
- [x] **Phase 3**: Gradio UI - Generate tab
- [x] **Phase 4**: Gradio UI - Train tab (recording, import)
- [ ] **Phase 5**: Gradio UI - Models tab
- [ ] **Phase 6**: Gradio UI - Settings tab
- [ ] **Phase 7**: Training backend
- [ ] **Phase 8**: Automation lanes

## License

MIT License
