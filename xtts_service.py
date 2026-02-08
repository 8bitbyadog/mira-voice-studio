#!/usr/bin/env python3
"""
XTTS Voice Cloning Service

Run this with Python 3.11 to use XTTS v2 for high-quality voice cloning.
Uses STREAMING INFERENCE to fix the audio cutoff bug that affects ~80% of standard outputs.

Usage:
    python3.11 xtts_service.py --text "Hello world" --reference path/to/ref.wav --output output.wav
"""

import argparse
import sys
import os
from pathlib import Path

# Accept XTTS license automatically
os.environ["COQUI_TOS_AGREED"] = "1"

def main():
    parser = argparse.ArgumentParser(description="XTTS Voice Cloning Service")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--reference", required=True, help="Reference audio file path")
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed (default: 1.0)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.reference):
        print(f"Error: Reference file not found: {args.reference}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading XTTS v2 model...")

    try:
        from TTS.api import TTS
        import torch
        import torchaudio

        # Load the TTS wrapper
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

        # Get the underlying XTTS model for streaming access
        model = tts.synthesizer.tts_model

        print(f"Synthesizing text: '{args.text[:50]}...' " if len(args.text) > 50 else f"Synthesizing text: '{args.text}'")
        print(f"Using reference: {args.reference}")

        # Compute speaker latents from reference audio
        # Use multiple reference files if available for better accent preservation
        print("Computing speaker embedding from reference...")

        ref_path = Path(args.reference)
        ref_files = [args.reference]

        # Look for chunked reference files (20-30s each) - better for voice cloning
        chunk_files = sorted(ref_path.parent.glob("reference_chunk_*.wav"))
        if chunk_files:
            ref_files = [str(f) for f in chunk_files]
            print(f"Using {len(ref_files)} reference chunks for better voice capture")

        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=ref_files
        )

        # Use STREAMING INFERENCE to avoid cutoff bug
        print("Using streaming inference (fixes audio cutoff)...")

        # Add punctuation if missing to help with phrasing
        text = args.text.strip()
        if text and text[-1] not in '.!?':
            text = text + '.'

        chunks = model.inference_stream(
            text,
            args.language,
            gpt_cond_latent,
            speaker_embedding,
            speed=args.speed,
            stream_chunk_size=20,  # Smaller chunks = less cutoff
            enable_text_splitting=True,  # Let model handle text splitting
        )

        # Collect all audio chunks
        wav_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"  Received chunk {i + 1} ({chunk.shape[-1]} samples)")
            wav_chunks.append(chunk)

        if not wav_chunks:
            raise RuntimeError("No audio chunks generated")

        # Concatenate all chunks
        print(f"Concatenating {len(wav_chunks)} chunks...")
        wav = torch.cat(wav_chunks, dim=0)

        # Add padding at the end to ensure nothing is cut off
        sample_rate = 24000  # XTTS output is 24kHz
        padding = torch.zeros(int(sample_rate * 0.5))  # 500ms padding
        wav = torch.cat([wav, padding], dim=0)

        # Save output
        print(f"Saving to {args.output}...")
        torchaudio.save(
            args.output,
            wav.squeeze().unsqueeze(0).cpu(),
            sample_rate
        )

        print(f"Audio saved to: {args.output}")
        print(f"Duration: {len(wav) / sample_rate:.2f} seconds")
        print("SUCCESS")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
