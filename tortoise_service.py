#!/usr/bin/env python3
"""
Tortoise-TTS Voice Cloning Service

Run this with Python 3.11 for high-quality voice cloning.
Slower than XTTS but often more natural sounding.

Usage:
    python3.11 tortoise_service.py --text "Hello world" --voice_dir path/to/voice --output output.wav
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Tortoise-TTS Voice Cloning Service")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice_dir", required=True, help="Directory containing reference WAV files")
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument("--preset", default="fast", choices=["ultra_fast", "fast", "standard", "high_quality"],
                        help="Quality preset (default: fast)")

    args = parser.parse_args()

    voice_dir = Path(args.voice_dir)
    if not voice_dir.exists():
        print(f"Error: Voice directory not found: {voice_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Tortoise-TTS...")

    try:
        import torch
        import torchaudio
        from tortoise.api import TextToSpeech
        from tortoise.utils.audio import load_voices

        # Initialize Tortoise
        tts = TextToSpeech()

        # Load voice samples from directory
        print(f"Loading voice samples from {voice_dir}...")

        # Get WAV files from the voice directory
        voice_samples = list(voice_dir.glob("*.wav"))
        if not voice_samples:
            # Check for references subdirectory
            refs_dir = voice_dir / "references"
            if refs_dir.exists():
                voice_samples = list(refs_dir.glob("*.wav"))

        if not voice_samples:
            print(f"Error: No WAV files found in {voice_dir}", file=sys.stderr)
            sys.exit(1)

        # Filter and select best samples:
        # - Skip very short clips (under 3 seconds)
        # - Sort by file size (larger = more audio content)
        # - Use top 5 samples for cleaner conditioning
        good_samples = []
        for s in voice_samples:
            try:
                info = torchaudio.info(str(s))
                duration = info.num_frames / info.sample_rate
                if duration >= 3.0:
                    good_samples.append((s, s.stat().st_size))
            except:
                pass

        good_samples.sort(key=lambda x: x[1], reverse=True)
        voice_samples = [s[0] for s in good_samples[:5]]
        print(f"Using {len(voice_samples)} best voice samples (3s+ duration)")

        # Load audio samples
        voice_clips = []
        for sample_path in voice_samples:
            audio, sr = torchaudio.load(str(sample_path))
            # Resample to 22050 if needed (Tortoise expects 22050)
            if sr != 22050:
                audio = torchaudio.functional.resample(audio, sr, 22050)
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            # Keep as 2D tensor [1, samples] - Tortoise expects this format
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            voice_clips.append(audio)

        print(f"Generating speech with preset: {args.preset}")
        print(f"Text: {args.text[:100]}..." if len(args.text) > 100 else f"Text: {args.text}")

        # Generate speech
        gen = tts.tts_with_preset(
            args.text,
            voice_samples=voice_clips,
            preset=args.preset,
        )

        # Save output
        print(f"Saving to {args.output}...")
        torchaudio.save(args.output, gen.squeeze(0).cpu(), 24000)

        print(f"Audio saved to: {args.output}")
        print("SUCCESS")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
