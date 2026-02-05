"""
Text processing for Mira Voice Studio.

Handles text input, sentence splitting, and chunk preparation.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Iterator
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize

from voice_studio.utils.slug import generate_slug, generate_chunk_filename


@dataclass
class Sentence:
    """Represents a single sentence to be processed."""

    index: int  # 1-based index
    text: str  # Original text
    slug: str  # Filename-safe version
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())

    @property
    def base_filename(self) -> str:
        """Get the base filename (without extension) for this sentence."""
        return f"{self.index:03d}_{self.slug}"

    def get_filename(self, extension: str) -> str:
        """Get filename with extension."""
        if not extension.startswith("."):
            extension = f".{extension}"
        return f"{self.base_filename}{extension}"


class TextProcessor:
    """
    Process text input into sentences for TTS generation.

    Handles:
    - Reading text from files or strings
    - Splitting text into sentences using NLTK
    - Cleaning and normalizing text
    - Generating metadata for each sentence
    """

    def __init__(self):
        """Initialize the text processor."""
        self._ensure_nltk_data()

    def _ensure_nltk_data(self) -> None:
        """Download required NLTK data if not present."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass  # Not all NLTK versions have punkt_tab

    def load_text(self, input_path: Path) -> str:
        """
        Load text from a file.

        Args:
            input_path: Path to the text file.

        Returns:
            The file contents as a string.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is empty.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        text = input_path.read_text(encoding="utf-8")

        if not text.strip():
            raise ValueError(f"Input file is empty: {input_path}")

        return text

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for TTS.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text.
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove excessive punctuation
        text = re.sub(r"([.!?])\1+", r"\1", text)

        # Normalize quotes
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")

        # Normalize dashes
        text = text.replace("—", " - ").replace("–", " - ")

        # Remove extra spaces around punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        # Clean the text first
        text = self.clean_text(text)

        # Use NLTK for sentence tokenization
        sentences = sent_tokenize(text)

        # Filter out empty sentences and clean each one
        cleaned = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned.append(sentence)

        return cleaned

    def process(
        self,
        text: str,
        source_name: Optional[str] = None
    ) -> List[Sentence]:
        """
        Process text into a list of Sentence objects.

        Args:
            text: Input text (string).
            source_name: Optional name for the source (used in logging).

        Returns:
            List of Sentence objects ready for TTS.
        """
        raw_sentences = self.split_sentences(text)

        sentences = []
        for i, text in enumerate(raw_sentences, start=1):
            slug = generate_slug(text, max_length=30)
            sentence = Sentence(
                index=i,
                text=text,
                slug=slug,
            )
            sentences.append(sentence)

        return sentences

    def process_file(self, input_path: Path) -> List[Sentence]:
        """
        Process a text file into sentences.

        Args:
            input_path: Path to the input text file.

        Returns:
            List of Sentence objects.
        """
        text = self.load_text(input_path)
        return self.process(text, source_name=input_path.name)

    def estimate_duration(
        self,
        sentences: List[Sentence],
        words_per_minute: float = 150,
        pause_ms: int = 300
    ) -> float:
        """
        Estimate total audio duration.

        Args:
            sentences: List of sentences.
            words_per_minute: Speaking rate.
            pause_ms: Pause between sentences in ms.

        Returns:
            Estimated duration in seconds.
        """
        total_words = sum(s.word_count for s in sentences)
        speaking_time = total_words / words_per_minute * 60
        pause_time = len(sentences) * pause_ms / 1000
        return speaking_time + pause_time

    def get_stats(self, sentences: List[Sentence]) -> dict:
        """
        Get statistics about the processed text.

        Args:
            sentences: List of sentences.

        Returns:
            Dictionary with statistics.
        """
        if not sentences:
            return {
                "sentence_count": 0,
                "word_count": 0,
                "avg_words_per_sentence": 0,
                "estimated_duration": "0:00",
            }

        word_count = sum(s.word_count for s in sentences)
        avg_words = word_count / len(sentences)
        duration = self.estimate_duration(sentences)

        # Format duration
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        duration_str = f"{minutes}:{seconds:02d}"

        return {
            "sentence_count": len(sentences),
            "word_count": word_count,
            "avg_words_per_sentence": round(avg_words, 1),
            "estimated_duration": duration_str,
            "estimated_duration_seconds": duration,
        }

    def iter_sentences(
        self,
        text: str,
        batch_size: Optional[int] = None
    ) -> Iterator[List[Sentence]]:
        """
        Iterate over sentences in batches.

        Useful for processing large texts without loading everything into memory.

        Args:
            text: Input text.
            batch_size: Number of sentences per batch. If None, yields all at once.

        Yields:
            Lists of Sentence objects.
        """
        sentences = self.process(text)

        if batch_size is None or batch_size <= 0:
            yield sentences
            return

        for i in range(0, len(sentences), batch_size):
            yield sentences[i:i + batch_size]
