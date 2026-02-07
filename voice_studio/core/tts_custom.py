"""
Custom voice TTS using trained reference audio.

Uses voice cloning to synthesize speech that matches the trained voice.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json

from voice_studio.core.tts_engine import TTSEngine, TTSError, ModelNotFoundError


class CustomVoiceTTS(TTSEngine):
    """
    TTS engine for custom trained voices.

    Uses the reference audio from training to clone the voice characteristics
    and apply them to synthesized speech.
    """

    def __init__(self, models_dir: Optional[Path] = None, sample_rate: int = 44100):
        """
        Initialize the custom voice TTS engine.

        Args:
            models_dir: Directory containing voice models.
            sample_rate: Output sample rate.
        """
        super().__init__(sample_rate=sample_rate)

        if models_dir is None:
            models_dir = Path.home() / "mira_voice_studio" / "models" / "custom"

        self.models_dir = Path(models_dir)
        self._ref_audio: Optional[np.ndarray] = None
        self._ref_audio_sr: int = 44100
        self._ref_text: Optional[str] = None
        self._voice_model_path: Optional[Path] = None
        self._xtts_model = None

    def load_voice(self, voice_name: str) -> None:
        """
        Load a custom trained voice.

        Args:
            voice_name: Name of the voice model to load.
        """
        model_dir = self.models_dir / voice_name

        if not model_dir.exists():
            raise ModelNotFoundError(f"Voice model not found: {voice_name}")

        # Look for combined reference first (better quality)
        combined_ref = model_dir / "reference_combined.wav"
        if combined_ref.exists():
            print(f"Using combined reference audio for better quality")
            self._load_reference_audio(combined_ref)
        else:
            # Find single reference audio
            ref_audio_files = (
                list(model_dir.glob("reference.wav")) +
                list(model_dir.glob("reference.mp3")) +
                list(model_dir.glob("ref_audio.*"))
            )

            if not ref_audio_files:
                raise ModelNotFoundError(f"No reference audio found for voice: {voice_name}")

            ref_path = ref_audio_files[0]
            self._load_reference_audio(ref_path)

        # Also load all individual references if available (for multi-reference extraction)
        refs_dir = model_dir / "references"
        if refs_dir.exists():
            self._all_reference_paths = sorted(refs_dir.glob("*.wav"))
            print(f"Found {len(self._all_reference_paths)} individual reference files")
        else:
            self._all_reference_paths = []

        # Load reference text
        ref_text_files = list(model_dir.glob("reference.txt"))
        if ref_text_files:
            self._ref_text = ref_text_files[0].read_text(encoding="utf-8").strip()

        # Store model info
        self._voice_model_path = model_dir
        self._current_voice = voice_name
        self._loaded = True

        # Try to initialize voice cloning backend
        self._init_voice_cloning()

    def _load_reference_audio(self, audio_path: Path) -> None:
        """Load reference audio file."""
        try:
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Store original for voice cloning
            self._ref_audio = audio.astype(np.float32)
            self._ref_audio_sr = sr
            self._ref_audio_path = audio_path

        except Exception as e:
            raise TTSError(f"Failed to load reference audio: {e}")

    def _init_voice_cloning(self) -> None:
        """Initialize voice cloning backend."""
        import shutil
        import subprocess

        # Try XTTS via Python 3.11 venv subprocess first (best quality)
        # Look for venv311 in the project directory
        project_root = Path(__file__).parent.parent.parent
        venv311_python = project_root / "venv311" / "bin" / "python"

        if venv311_python.exists():
            # Check if TTS is installed in venv311
            try:
                result = subprocess.run(
                    [str(venv311_python), "-c", "from TTS.api import TTS; print('ok')"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0 and "ok" in result.stdout:
                    self._python311_path = str(venv311_python)
                    self._backend = "xtts_subprocess"
                    print("Using XTTS v2 via Python 3.11 for high-quality voice cloning")
                    return
            except Exception as e:
                print(f"XTTS subprocess check failed: {e}")

        # Try OpenVoice CLI (works on Python 3.12)
        try:
            from openvoice_cli.api import ToneColorConverter
            import torch

            # Determine device
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            self._openvoice_device = device
            self._backend = "openvoice"
            print(f"Using OpenVoice for voice cloning (device: {device})")
            return
        except ImportError as e:
            print(f"OpenVoice import failed: {e}")
        except Exception as e:
            print(f"OpenVoice not available: {e}")

        # Try Coqui XTTS directly (if Python version supports it)
        try:
            from TTS.api import TTS
            self._xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._backend = "xtts"
            print("Using XTTS v2 for voice cloning")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"XTTS not available: {e}")

        # Fallback: we'll use pitch/timbre transfer
        self._backend = "transfer"
        print("Using audio transfer for voice cloning")

    def synthesize(self, text: str, speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech in the custom voice.

        Args:
            text: Text to synthesize.
            speed: Speaking speed multiplier.

        Returns:
            Tuple of (audio_data, sample_rate).
        """
        if not self._loaded:
            raise TTSError("No voice loaded. Call load_voice() first.")

        if not text.strip():
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32), self.sample_rate

        try:
            if self._backend == "xtts_subprocess":
                return self._synthesize_xtts_subprocess(text, speed)
            elif self._backend == "openvoice":
                return self._synthesize_openvoice(text, speed)
            elif self._backend == "xtts" and self._xtts_model is not None:
                return self._synthesize_xtts(text, speed)
            else:
                return self._synthesize_transfer(text, speed)
        except Exception as e:
            raise TTSError(f"Synthesis failed: {e}")

    def _synthesize_xtts_subprocess(self, text: str, speed: float) -> Tuple[np.ndarray, int]:
        """Synthesize using XTTS v2 via Python 3.11 subprocess."""
        import tempfile
        import subprocess
        import soundfile as sf
        import os

        # Create temp output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            # Get reference audio path
            ref_path = str(self._ref_audio_path)

            # Find the xtts_service.py script
            service_script = Path(__file__).parent.parent.parent / "xtts_service.py"
            if not service_script.exists():
                raise FileNotFoundError(f"XTTS service script not found at {service_script}")

            print(f"Running XTTS synthesis via Python 3.11...")

            # Call XTTS via Python 3.11
            env = os.environ.copy()
            env["COQUI_TOS_AGREED"] = "1"
            result = subprocess.run(
                [
                    self._python311_path,
                    str(service_script),
                    "--text", text,
                    "--reference", ref_path,
                    "--output", output_path,
                    "--speed", str(speed),
                    "--language", "en"
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env
            )

            if result.returncode != 0:
                print(f"XTTS stderr: {result.stderr}")
                raise RuntimeError(f"XTTS synthesis failed: {result.stderr}")

            if "SUCCESS" not in result.stdout:
                print(f"XTTS stdout: {result.stdout}")
                raise RuntimeError(f"XTTS synthesis did not complete successfully")

            # Load the output audio
            audio, sr = sf.read(output_path)

            # Resample if needed
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)

            return audio.astype(np.float32), self.sample_rate

        finally:
            # Cleanup temp file
            Path(output_path).unlink(missing_ok=True)

    def _synthesize_openvoice(self, text: str, speed: float) -> Tuple[np.ndarray, int]:
        """Synthesize using OpenVoice voice cloning."""
        import tempfile
        import soundfile as sf
        from pathlib import Path

        try:
            from openvoice_cli.api import ToneColorConverter
            import torch

            # Create temp directory for processing
            temp_dir = Path(tempfile.mkdtemp(prefix="openvoice_"))

            try:
                # Step 1: Generate base audio with Edge TTS
                from voice_studio.core.tts_edge import EdgeTTS
                edge_tts = EdgeTTS(sample_rate=22050)  # OpenVoice uses 22050Hz
                edge_tts.load_voice("en-US-GuyNeural")  # Neutral base voice
                base_audio, base_sr = edge_tts.synthesize(text, speed)
                edge_tts.unload()

                # Save base audio
                base_path = temp_dir / "base.wav"
                sf.write(str(base_path), base_audio, base_sr)

                # Save reference audio (resample to 22050 if needed)
                ref_path = temp_dir / "reference.wav"
                if self._ref_audio_sr != 22050:
                    ref_audio_resampled = self._resample(self._ref_audio, self._ref_audio_sr, 22050)
                else:
                    ref_audio_resampled = self._ref_audio
                sf.write(str(ref_path), ref_audio_resampled, 22050)

                # Step 2: Load OpenVoice tone color converter
                home = Path.home()
                openvoice_dir = home / ".cache" / "openvoice"
                config_path = openvoice_dir / "converter" / "config.json"
                ckpt_path = openvoice_dir / "converter" / "checkpoint.pth"

                if not config_path.exists() or not ckpt_path.exists():
                    raise FileNotFoundError("OpenVoice models not found. Run model download first.")

                # Use CPU for MPS compatibility (MPS has issues with some ops)
                device = "cpu"  # self._openvoice_device

                tone_color_converter = ToneColorConverter(
                    str(config_path),
                    device=device
                )
                tone_color_converter.load_ckpt(str(ckpt_path))

                # Step 3: Extract speaker embedding from reference(s)
                # Use multiple references if available for better quality
                if hasattr(self, '_all_reference_paths') and len(self._all_reference_paths) > 1:
                    # Use up to 10 best references
                    ref_paths = [str(p) for p in self._all_reference_paths[:10]]
                    print(f"Extracting speaker embedding from {len(ref_paths)} references...")
                    target_se = tone_color_converter.extract_se(ref_paths)
                else:
                    target_se = tone_color_converter.extract_se(str(ref_path))

                # Step 4: Extract source embedding from base audio
                source_se = tone_color_converter.extract_se(str(base_path))

                # Step 5: Convert tone color
                output_path = temp_dir / "output.wav"
                tone_color_converter.convert(
                    audio_src_path=str(base_path),
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=str(output_path),
                    message="AutoVoice"
                )

                # Load result
                audio, sr = sf.read(str(output_path))

                # Resample to target sample rate
                if sr != self.sample_rate:
                    audio = self._resample(audio, sr, self.sample_rate)

                return audio.astype(np.float32), self.sample_rate

            finally:
                # Cleanup temp files
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            print(f"OpenVoice failed: {e}, falling back to transfer")
            import traceback
            traceback.print_exc()
            return self._synthesize_transfer(text, speed)

    def _synthesize_xtts(self, text: str, speed: float) -> Tuple[np.ndarray, int]:
        """Synthesize using XTTS voice cloning."""
        import tempfile
        import soundfile as sf

        # XTTS needs reference audio as a file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref_path = f.name
            sf.write(ref_path, self._ref_audio, self._ref_audio_sr)

        try:
            # Generate with voice cloning
            output_path = tempfile.mktemp(suffix=".wav")
            self._xtts_model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=ref_path,
                language="en",
                speed=speed
            )

            # Load result
            audio, sr = sf.read(output_path)

            # Cleanup
            Path(ref_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

            # Resample if needed
            if sr != self.sample_rate:
                audio = self._resample(audio, sr, self.sample_rate)

            return audio.astype(np.float32), self.sample_rate

        finally:
            Path(ref_path).unlink(missing_ok=True)

    def _synthesize_transfer(self, text: str, speed: float) -> Tuple[np.ndarray, int]:
        """
        Synthesize using Edge TTS + voice characteristic transfer.

        This generates base audio with Edge TTS, then applies pitch and
        timbre characteristics from the reference audio.
        """
        from voice_studio.core.tts_edge import EdgeTTS

        # Generate base audio with Edge TTS
        edge_tts = EdgeTTS(sample_rate=self.sample_rate)
        edge_tts.load_voice("en-US-GuyNeural")  # Use a neutral voice as base
        base_audio, sr = edge_tts.synthesize(text, speed)
        edge_tts.unload()

        # Apply voice characteristics from reference
        output_audio = self._apply_voice_transfer(base_audio, sr)

        return output_audio, self.sample_rate

    def _apply_voice_transfer(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply voice characteristics from reference audio to synthesized audio.

        Uses pitch shifting and formant adjustment to match the reference voice.
        """
        try:
            import librosa

            # Analyze reference voice characteristics
            ref_audio_resampled = librosa.resample(
                self._ref_audio,
                orig_sr=self._ref_audio_sr,
                target_sr=sr
            )

            # Get pitch info from reference
            ref_pitches, ref_magnitudes = librosa.piptrack(
                y=ref_audio_resampled,
                sr=sr,
                fmin=50,
                fmax=500
            )
            ref_pitch_mean = np.mean(ref_pitches[ref_pitches > 0]) if np.any(ref_pitches > 0) else 150

            # Get pitch info from generated audio
            gen_pitches, gen_magnitudes = librosa.piptrack(
                y=audio,
                sr=sr,
                fmin=50,
                fmax=500
            )
            gen_pitch_mean = np.mean(gen_pitches[gen_pitches > 0]) if np.any(gen_pitches > 0) else 150

            # Calculate pitch shift needed (in semitones)
            if gen_pitch_mean > 0 and ref_pitch_mean > 0:
                pitch_shift = 12 * np.log2(ref_pitch_mean / gen_pitch_mean)
                pitch_shift = np.clip(pitch_shift, -12, 12)  # Limit to 1 octave
            else:
                pitch_shift = 0

            # Apply pitch shift if significant
            if abs(pitch_shift) > 0.5:
                audio = librosa.effects.pitch_shift(
                    audio,
                    sr=sr,
                    n_steps=pitch_shift
                )

            return audio.astype(np.float32)

        except ImportError:
            # librosa not available, return unmodified
            print("Warning: librosa not available, skipping voice transfer")
            return audio
        except Exception as e:
            print(f"Voice transfer failed: {e}, using original audio")
            return audio

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio

        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback to scipy
            from scipy import signal
            new_length = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, new_length).astype(np.float32)

    def unload(self) -> None:
        """Unload the current voice."""
        self._ref_audio = None
        self._ref_text = None
        self._voice_model_path = None
        self._current_voice = None
        self._loaded = False

        if self._xtts_model is not None:
            del self._xtts_model
            self._xtts_model = None

        import gc
        gc.collect()

    def list_voices(self) -> List[str]:
        """List available custom voices."""
        voices = []
        if self.models_dir.exists():
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    # Check for reference audio
                    has_ref = (
                        any(model_dir.glob("reference.*")) or
                        any(model_dir.glob("ref_audio.*"))
                    )
                    if has_ref:
                        voices.append(model_dir.name)
        return voices
