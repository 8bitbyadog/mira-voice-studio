"""
Slug generation for filename-safe identifiers.

Converts text into clean, filesystem-safe slugs for chunk naming.
"""

import re
import unicodedata
from typing import Optional


def generate_slug(
    text: str,
    max_length: int = 30,
    separator: str = "_"
) -> str:
    """
    Generate a filename-safe slug from text.

    Converts text to lowercase, removes punctuation, replaces spaces
    with underscores, and truncates to max_length.

    Args:
        text: The input text to convert.
        max_length: Maximum length of the resulting slug.
        separator: Character to use between words (default: underscore).

    Returns:
        A clean, filesystem-safe slug.

    Examples:
        >>> generate_slug("Welcome to my channel, everyone!")
        'welcome_to_my_channel_everyone'

        >>> generate_slug("Today we're going to talk about something.")
        'today_were_going_to_talk_about'

        >>> generate_slug("What's up?!")
        'whats_up'
    """
    if not text:
        return "untitled"

    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Convert to lowercase
    text = text.lower()

    # Remove contractions' apostrophes but keep the letters
    text = re.sub(r"'", "", text)

    # Replace any non-alphanumeric character with separator
    text = re.sub(r"[^a-z0-9]+", separator, text)

    # Remove leading/trailing separators
    text = text.strip(separator)

    # Collapse multiple separators
    text = re.sub(f"{separator}+", separator, text)

    # Truncate to max_length, but don't cut in middle of a word
    if len(text) > max_length:
        # Find the last separator before max_length
        truncated = text[:max_length]
        last_sep = truncated.rfind(separator)
        if last_sep > max_length // 2:
            text = truncated[:last_sep]
        else:
            text = truncated

    # Remove trailing separator after truncation
    text = text.rstrip(separator)

    return text if text else "untitled"


def generate_chunk_filename(
    index: int,
    text: str,
    extension: str = "",
    max_slug_length: int = 30
) -> str:
    """
    Generate a chunk filename with index prefix and slug.

    Args:
        index: The chunk index (1-based).
        text: The sentence text to convert to slug.
        extension: File extension (with or without dot).
        max_slug_length: Maximum length for the slug portion.

    Returns:
        Filename in format: {index:03d}_{slug}.{extension}

    Examples:
        >>> generate_chunk_filename(1, "Welcome to my channel!", "wav")
        '001_welcome_to_my_channel.wav'

        >>> generate_chunk_filename(42, "This is a test.", "srt")
        '042_this_is_a_test.srt'
    """
    slug = generate_slug(text, max_length=max_slug_length)

    # Ensure extension starts with dot if provided
    if extension and not extension.startswith("."):
        extension = f".{extension}"

    return f"{index:03d}_{slug}{extension}"


def generate_session_name(prefix: str = "session") -> str:
    """
    Generate a session name with timestamp.

    Args:
        prefix: Prefix for the session name.

    Returns:
        Session name in format: {prefix}_{YYYY-MM-DD_HHMMSS}
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def generate_output_folder_name(source_name: str) -> str:
    """
    Generate an output folder name with date prefix.

    Args:
        source_name: The source file or project name.

    Returns:
        Folder name in format: {YYYY-MM-DD}_{slug}

    Examples:
        >>> generate_output_folder_name("My Amazing Script.txt")
        '2024-02-05_my_amazing_script'
    """
    from datetime import datetime

    # Remove file extension if present
    if "." in source_name:
        source_name = source_name.rsplit(".", 1)[0]

    date_prefix = datetime.now().strftime("%Y-%m-%d")
    slug = generate_slug(source_name, max_length=40)

    return f"{date_prefix}_{slug}"
