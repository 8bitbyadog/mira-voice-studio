"""
Output manager for Mira Voice Studio.

Handles creating output folder structures and saving all generated files.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
import shutil

from voice_studio.core.text_processor import TextProcessor, Sentence
from voice_studio.core.tts_engine import TTSEngine, TTSResult
from voice_studio.core.audio_stitcher import AudioStitcher, StitchedAudio, ChunkTiming
from voice_studio.core.aligner import WhisperAligner, AlignmentResult
from voice_studio.core.srt_generator import SRTGenerator
from voice_studio.core.manifest import ManifestGenerator, Manifest, ManifestSettings
from voice_studio.utils.slug import generate_output_folder_name
from voice_studio.utils.audio_utils import save_audio


@dataclass
class GenerationResult:
    """Result of a complete generation session."""

    output_dir: Path
    master_audio: Path
    master_srt: Path
    master_srt_words: Optional[Path]
    master_vtt: Path
    manifest_path: Path
    manifest: Manifest
    stitched_audio: StitchedAudio
    alignment: Optional[AlignmentResult]

    @property
    def success_rate(self) -> float:
        """Get the success rate as a percentage."""
        total = self.manifest.total_sentences
        if total == 0:
            return 0.0
        return (self.manifest.successful_sentences / total) * 100

    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"Generated {self.manifest.successful_sentences}/{self.manifest.total_sentences} "
            f"sentences ({self.success_rate:.0f}% success)\n"
            f"Duration: {self.manifest.total_duration_seconds:.1f}s\n"
            f"Output: {self.output_dir}"
        )


class OutputManager:
    """
    Manage output file creation and organization.

    Handles:
    - Creating output folder structure
    - Saving master audio and captions
    - Exporting individual chunks (on demand)
    - Exporting I/O selections (on demand)
    - Creating and updating manifest.json
    """

    def __init__(
        self,
        base_output_dir: Optional[Path] = None,
        sample_rate: int = 44100,
        pause_ms: int = 300
    ):
        """
        Initialize the output manager.

        Args:
            base_output_dir: Base directory for outputs.
            sample_rate: Audio sample rate.
            pause_ms: Pause between sentences in ms.
        """
        if base_output_dir is None:
            base_output_dir = Path.home() / "Videos" / "VO"

        self.base_output_dir = Path(base_output_dir)
        self.sample_rate = sample_rate
        self.pause_ms = pause_ms

        # Sub-components
        self.text_processor = TextProcessor()
        self.stitcher = AudioStitcher(pause_ms=pause_ms, sample_rate=sample_rate)
        self.srt_generator = SRTGenerator()
        self.manifest_generator = ManifestGenerator()

    def create_output_folder(
        self,
        source_name: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Create the output folder structure.

        Args:
            source_name: Name of the source file/project.
            output_dir: Override output directory.

        Returns:
            Path to the created output folder.
        """
        if output_dir is None:
            folder_name = generate_output_folder_name(source_name)
            output_dir = self.base_output_dir / folder_name

        output_dir = Path(output_dir)
        master_dir = output_dir / "master"

        # Create directories
        output_dir.mkdir(parents=True, exist_ok=True)
        master_dir.mkdir(exist_ok=True)

        return output_dir

    def generate(
        self,
        text: str,
        source_name: str,
        tts_engine: TTSEngine,
        aligner: Optional[WhisperAligner] = None,
        speed: float = 1.0,
        output_dir: Optional[Path] = None,
        generate_word_captions: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> GenerationResult:
        """
        Run the complete generation pipeline.

        Args:
            text: Input text to generate.
            source_name: Name for the output folder.
            tts_engine: TTS engine to use.
            aligner: Whisper aligner for word-level captions.
            speed: Speaking speed.
            output_dir: Override output directory.
            generate_word_captions: Whether to generate word-level captions.
            progress_callback: Optional callback(stage, current, total).

        Returns:
            GenerationResult with all output information.
        """
        # Create output folder
        output_dir = self.create_output_folder(source_name, output_dir)
        master_dir = output_dir / "master"

        # Get base name for files
        base_name = generate_output_folder_name(source_name).split("_", 1)[-1]
        if not base_name:
            base_name = "output"

        # Step 1: Process text into sentences
        if progress_callback:
            progress_callback("Processing text", 0, 1)

        sentences = self.text_processor.process(text, source_name)

        # Step 2: Generate TTS for each sentence
        def tts_progress(current: int, total: int, sentence: Sentence):
            if progress_callback:
                progress_callback(f"Generating sentence {current}/{total}", current, total)

        results = tts_engine.generate_batch(sentences, speed=speed, progress_callback=tts_progress)

        # Step 3: Stitch audio together
        if progress_callback:
            progress_callback("Stitching audio", 0, 1)

        stitched = self.stitcher.stitch(results)

        # Step 4: Save master audio
        if progress_callback:
            progress_callback("Saving audio", 0, 1)

        master_audio_path = master_dir / f"{base_name}.wav"
        save_audio(master_audio_path, stitched.audio_data, stitched.sample_rate)

        # Step 5: Run alignment for word-level timestamps
        alignment = None
        if generate_word_captions and aligner is not None:
            if progress_callback:
                progress_callback("Aligning audio", 0, 1)
            try:
                alignment = aligner.align(stitched.audio_data, stitched.sample_rate)
            except Exception as e:
                print(f"Warning: Alignment failed: {e}")

        # Step 6: Generate captions
        if progress_callback:
            progress_callback("Generating captions", 0, 1)

        caption_files = self.srt_generator.generate_from_chunk_timings(
            chunk_timings=stitched.chunk_timings,
            output_dir=master_dir,
            base_name=base_name,
            generate_word_level=generate_word_captions and alignment is not None,
            alignment=alignment,
        )

        # Step 7: Create manifest
        if progress_callback:
            progress_callback("Creating manifest", 0, 1)

        master_files = {
            "audio": master_audio_path,
            "srt_sentences": caption_files.get("srt_sentences"),
            "srt_words": caption_files.get("srt_words"),
            "vtt": caption_files.get("vtt"),
        }

        settings = ManifestSettings(
            pause_between_sentences_ms=self.pause_ms,
            sample_rate=self.sample_rate,
            bit_depth=16,
            word_level_captions=generate_word_captions and alignment is not None,
        )

        manifest = self.manifest_generator.create_manifest(
            source_name=source_name,
            voice=tts_engine.current_voice or "default",
            speed=speed,
            chunk_timings=stitched.chunk_timings,
            master_files=master_files,
            output_dir=output_dir,
            settings=settings,
        )

        manifest_path = output_dir / "manifest.json"
        self.manifest_generator.save_manifest(manifest, manifest_path)

        if progress_callback:
            progress_callback("Complete", 1, 1)

        return GenerationResult(
            output_dir=output_dir,
            master_audio=master_audio_path,
            master_srt=caption_files.get("srt_sentences"),
            master_srt_words=caption_files.get("srt_words"),
            master_vtt=caption_files.get("vtt"),
            manifest_path=manifest_path,
            manifest=manifest,
            stitched_audio=stitched,
            alignment=alignment,
        )

    def export_chunks(
        self,
        result: GenerationResult,
        tts_results: List[TTSResult]
    ) -> Path:
        """
        Export individual sentence chunks.

        Args:
            result: GenerationResult from generate().
            tts_results: Original TTS results with audio data.

        Returns:
            Path to the chunks directory.
        """
        chunks_dir = result.output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        # Save each chunk's audio and SRT
        for tts_result, timing in zip(tts_results, result.stitched_audio.chunk_timings):
            if not tts_result.success:
                continue

            sentence = tts_result.sentence

            # Save audio
            audio_path = chunks_dir / sentence.get_filename("wav")
            save_audio(audio_path, tts_result.audio_data, tts_result.sample_rate)

            # Save SRT (timestamps start at 0)
            srt_path = chunks_dir / sentence.get_filename("srt")
            self.srt_generator.generate_chunk_srt(timing, srt_path)

            # Save plain text
            txt_path = chunks_dir / sentence.get_filename("txt")
            txt_path.write_text(sentence.text, encoding="utf-8")

        # Update manifest
        self.manifest_generator.update_manifest_chunks_exported(
            result.manifest_path,
            exported=True
        )

        return chunks_dir

    def get_output_info(self, output_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Get information about an existing output directory.

        Args:
            output_dir: Path to the output directory.

        Returns:
            Dictionary with output information, or None if not found.
        """
        output_dir = Path(output_dir)
        manifest_path = output_dir / "manifest.json"

        if not manifest_path.exists():
            return None

        try:
            manifest = self.manifest_generator.load_manifest(manifest_path)
            return {
                "output_dir": output_dir,
                "manifest": manifest,
                "has_chunks": (output_dir / "chunks").exists(),
                "has_selections": (output_dir / "selections").exists(),
            }
        except Exception:
            return None
