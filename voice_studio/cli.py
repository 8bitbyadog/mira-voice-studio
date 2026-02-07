#!/usr/bin/env python3
"""
Auto Voice - Command Line Interface

Generate voiceovers with synced captions from text input.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="voice_studio",
        description="Auto Voice - Generate voiceovers with synced captions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Basic — generates master only
  voice_studio -i script.txt

  # With all chunks
  voice_studio -i script.txt --export-chunks

  # Export just a selection
  voice_studio -i script.txt --in-time 0:32 --out-time 1:45

  # Specify output folder
  voice_studio -i script.txt -o ~/Desktop/my_video/

  # Use specific Coqui voice
  voice_studio -i script.txt --voice vits --speed 1.1

  # Use GPT-SoVITS with trained voice
  voice_studio -i script.txt --engine gptsovits --voice my_voice

  # Use GPT-SoVITS with reference audio (zero-shot)
  voice_studio -i script.txt --engine gptsovits \\
    --ref-audio reference.wav --ref-text "Hello, this is my voice."

  # List available voices
  voice_studio --list-voices

  # Launch web UI
  voice_studio --ui
  voice_studio --ui --port 8080
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input text file or quoted string",
    )

    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (default: ~/Videos/VO/)",
    )

    parser.add_argument(
        "--voice", "-v",
        type=str,
        default="default",
        help="Voice model name (default: default)",
    )

    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Speaking speed 0.5-2.0 (default: 1.0)",
    )

    parser.add_argument(
        "--pause", "-p",
        type=int,
        default=300,
        help="Pause between sentences in ms (default: 300)",
    )

    parser.add_argument(
        "--word-level", "-w",
        action="store_true",
        help="Include word-level captions",
    )

    parser.add_argument(
        "--export-chunks",
        action="store_true",
        help="Also export individual sentence chunks",
    )

    parser.add_argument(
        "--in-time",
        type=str,
        default=None,
        help="Export selection start time (e.g., '0:32' or '32')",
    )

    parser.add_argument(
        "--out-time",
        type=str,
        default=None,
        help="Export selection end time (e.g., '1:45' or '105')",
    )

    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voice models and exit",
    )

    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for alignment (default: base)",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--engine", "-e",
        type=str,
        choices=["coqui", "gptsovits", "auto"],
        default="auto",
        help="TTS engine to use (default: auto)",
    )

    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio file for GPT-SoVITS voice cloning",
    )

    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of reference audio for GPT-SoVITS",
    )

    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the web UI instead of CLI mode",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web UI (default: 7860)",
    )

    return parser


def list_voices():
    """List available voice models."""
    from voice_studio.core.tts_coqui import CoquiTTS
    from voice_studio.core.tts_gptsovits import GPTSoVITS

    print("\n" + "=" * 50)
    print("Available Voice Models")
    print("=" * 50)

    # Coqui TTS voices
    print("\nCoqui TTS (--engine coqui):")
    print("-" * 40)
    try:
        coqui = CoquiTTS()
        coqui_voices = coqui.list_voices()
        for voice in coqui_voices:
            print(f"  • {voice}")
    except Exception as e:
        print(f"  (Error loading Coqui: {e})")

    # GPT-SoVITS voices
    print("\nGPT-SoVITS (--engine gptsovits):")
    print("-" * 40)
    try:
        gptsovits = GPTSoVITS()
        gptsovits_voices = gptsovits.list_voices()
        if gptsovits_voices:
            for voice in gptsovits_voices:
                print(f"  • {voice}")
        else:
            print("  (No custom voices found)")
            print("  Add voices to: ~/mira_voice_studio/models/custom/")
    except Exception as e:
        print(f"  (Error loading GPT-SoVITS: {e})")

    print("\n" + "-" * 40)
    print("Usage:")
    print("  voice_studio -i script.txt --voice default")
    print("  voice_studio -i script.txt --engine gptsovits --voice my_voice")
    print("\nFor GPT-SoVITS with reference audio:")
    print("  voice_studio -i script.txt --engine gptsovits \\")
    print("    --ref-audio reference.wav --ref-text 'Hello world'")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle --ui (launch web interface)
    if args.ui:
        from voice_studio.ui import launch
        print("Launching Auto Voice Web UI...")
        launch(server_port=args.port, debug=False)
        sys.exit(0)

    # Handle --list-voices
    if args.list_voices:
        list_voices()
        sys.exit(0)

    # Validate input
    if not args.input:
        parser.error("--input/-i is required (or use --ui for web interface)")

    # Import here to avoid slow startup for --help
    from voice_studio.core.tts_coqui import CoquiTTS
    from voice_studio.core.tts_gptsovits import GPTSoVITS
    from voice_studio.core.aligner import WhisperAligner
    from voice_studio.core.output_manager import OutputManager
    from voice_studio.core.selection import Selection, SelectionExporter
    from voice_studio.utils.device import get_device_info

    # Show device info
    if not args.quiet:
        device_info = get_device_info()
        print(f"\nDevice: {device_info['device']}")
        print(f"PyTorch: {device_info['pytorch_version']}")

    # Determine input text
    input_path = Path(args.input)
    if input_path.exists():
        text = input_path.read_text(encoding="utf-8")
        source_name = input_path.stem
        if not args.quiet:
            print(f"Input: {input_path}")
    else:
        # Treat as literal text
        text = args.input
        source_name = "script"
        if not args.quiet:
            print(f"Input: (text, {len(text)} characters)")

    if not text.strip():
        print("Error: Input text is empty")
        sys.exit(1)

    # Determine which TTS engine to use
    engine = args.engine
    if engine == "auto":
        # Use GPT-SoVITS if ref-audio provided or voice looks like a custom model
        if args.ref_audio or "/" in args.voice or args.voice not in ["default", "fast", "vits", "vits_neon"]:
            # Check if GPT-SoVITS has this voice
            try:
                gptsovits = GPTSoVITS()
                if args.voice in gptsovits.list_voices():
                    engine = "gptsovits"
                else:
                    engine = "coqui"
            except Exception:
                engine = "coqui"
        else:
            engine = "coqui"

    # Initialize TTS engine
    if not args.quiet:
        print(f"\nLoading TTS engine: {engine} (voice: {args.voice})...")

    try:
        if engine == "gptsovits":
            tts = GPTSoVITS()

            # Set reference audio if provided
            if args.ref_audio:
                ref_audio_path = Path(args.ref_audio)
                if not ref_audio_path.exists():
                    print(f"Error: Reference audio not found: {args.ref_audio}")
                    sys.exit(1)
                tts.set_reference(
                    audio_path=ref_audio_path,
                    text=args.ref_text
                )
                if not args.quiet:
                    print(f"  Reference audio: {ref_audio_path}")
                # Mark as loaded for direct reference use
                tts._loaded = True
                tts._current_voice = "custom_reference"
            else:
                tts.load_voice(args.voice)
        else:
            tts = CoquiTTS()
            tts.load_voice(args.voice)

    except Exception as e:
        print(f"Error loading voice '{args.voice}': {e}")
        sys.exit(1)

    # Initialize aligner if word-level captions requested
    aligner = None
    if args.word_level:
        if not args.quiet:
            print(f"Loading Whisper ({args.whisper_model}) for word-level alignment...")
        aligner = WhisperAligner(model_size=args.whisper_model)

    # Determine output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)

    # Initialize output manager
    output_manager = OutputManager(
        base_output_dir=output_dir.parent if output_dir else None,
        sample_rate=44100,
        pause_ms=args.pause,
    )

    # Progress callback
    pbar = None

    def progress_callback(stage: str, current: int, total: int):
        nonlocal pbar
        if args.quiet:
            return

        if "Generating sentence" in stage:
            if pbar is None:
                pbar = tqdm(total=total, desc="Generating", unit="sentence")
            pbar.update(1)
        else:
            if pbar is not None:
                pbar.close()
                pbar = None
            print(f"  {stage}...")

    # Run generation
    if not args.quiet:
        print("\nGenerating voiceover...")

    try:
        result = output_manager.generate(
            text=text,
            source_name=source_name,
            tts_engine=tts,
            aligner=aligner,
            speed=args.speed,
            output_dir=output_dir,
            generate_word_captions=args.word_level,
            progress_callback=progress_callback if not args.quiet else None,
        )
    except Exception as e:
        print(f"\nError during generation: {e}")
        sys.exit(1)
    finally:
        if pbar:
            pbar.close()

    # Export chunks if requested
    if args.export_chunks:
        if not args.quiet:
            print("\nExporting individual chunks...")

        # Need to re-generate to get TTS results for chunks
        from voice_studio.core.text_processor import TextProcessor
        processor = TextProcessor()
        sentences = processor.process(text)
        tts_results = tts.generate_batch(sentences, speed=args.speed)

        chunks_dir = output_manager.export_chunks(result, tts_results)
        if not args.quiet:
            print(f"  Chunks: {chunks_dir}")

    # Export selection if requested
    if args.in_time and args.out_time:
        if not args.quiet:
            print("\nExporting selection...")

        try:
            in_time = Selection.from_string(args.in_time)
            out_time = Selection.from_string(args.out_time)
            selection = Selection(in_time=in_time, out_time=out_time)

            exporter = SelectionExporter()
            selection_dir = exporter.export(
                selection=selection,
                master_audio=result.stitched_audio.audio_data,
                sample_rate=result.stitched_audio.sample_rate,
                chunk_timings=result.stitched_audio.chunk_timings,
                output_dir=result.output_dir,
                manifest_path=result.manifest_path,
            )

            if not args.quiet:
                in_tc, out_tc = selection.to_timecode()
                print(f"  Selection ({in_tc} → {out_tc}): {selection_dir}")

        except ValueError as e:
            print(f"Error with selection times: {e}")

    # Summary
    if not args.quiet:
        print("\n" + "=" * 50)
        print("GENERATION COMPLETE")
        print("=" * 50)
        print(result.summary())
        print("\nOutput files:")
        print(f"  Audio: {result.master_audio}")
        print(f"  SRT:   {result.master_srt}")
        if result.master_srt_words:
            print(f"  Words: {result.master_srt_words}")
        print(f"  VTT:   {result.master_vtt}")

    # Cleanup
    tts.unload()
    if aligner:
        aligner.unload_model()


if __name__ == "__main__":
    main()
