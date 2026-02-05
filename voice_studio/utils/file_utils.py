"""
File utilities for Mira Voice Studio.

Includes macOS Finder integration and file management helpers.
"""

import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List


def open_in_finder(path: Path) -> bool:
    """
    Open a file or folder in macOS Finder.

    Args:
        path: Path to the file or folder to reveal.

    Returns:
        True if successful, False otherwise.
    """
    try:
        path = Path(path).resolve()
        if path.is_file():
            # Reveal file in Finder
            subprocess.run(["open", "-R", str(path)], check=True)
        else:
            # Open folder in Finder
            subprocess.run(["open", str(path)], check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to the macOS clipboard.

    Args:
        text: Text to copy to clipboard.

    Returns:
        True if successful, False otherwise.
    """
    try:
        subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def get_safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Convert a string to a safe filename.

    Args:
        filename: The proposed filename.
        max_length: Maximum allowed filename length.

    Returns:
        A filesystem-safe filename.
    """
    # Remove or replace invalid characters
    # macOS allows most characters except : and /
    filename = re.sub(r'[:/]', '_', filename)

    # Remove null bytes and other control characters
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)

    # Remove leading/trailing whitespace and dots
    filename = filename.strip(' .')

    # Truncate if too long (preserving extension)
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext

    # Default if empty
    if not filename:
        filename = "untitled"

    return filename


def ensure_parent_exists(path: Path) -> Path:
    """
    Ensure the parent directory of a path exists.

    Args:
        path: Path whose parent directory should exist.

    Returns:
        The original path (for chaining).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_unique_path(path: Path) -> Path:
    """
    Get a unique path by appending a number if the path exists.

    Args:
        path: The desired path.

    Returns:
        A path that doesn't exist (may be the original if unique).

    Examples:
        If 'output.wav' exists, returns 'output_2.wav', etc.
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 2
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def list_audio_files(directory: Path, recursive: bool = False) -> List[Path]:
    """
    List all audio files in a directory.

    Args:
        directory: Directory to search.
        recursive: Whether to search subdirectories.

    Returns:
        List of paths to audio files.
    """
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

    directory = Path(directory)
    if not directory.is_dir():
        return []

    if recursive:
        files = directory.rglob("*")
    else:
        files = directory.glob("*")

    return sorted([
        f for f in files
        if f.is_file() and f.suffix.lower() in audio_extensions
    ])


def get_file_size_human(path: Path) -> str:
    """
    Get human-readable file size.

    Args:
        path: Path to the file.

    Returns:
        Human-readable size string (e.g., "4.2 MB").
    """
    try:
        size = Path(path).stat().st_size
    except (OSError, FileNotFoundError):
        return "0 B"

    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def cleanup_temp_files(directory: Path, pattern: str = "*.tmp") -> int:
    """
    Remove temporary files matching a pattern.

    Args:
        directory: Directory to clean.
        pattern: Glob pattern for files to remove.

    Returns:
        Number of files removed.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0

    count = 0
    for file in directory.glob(pattern):
        try:
            file.unlink()
            count += 1
        except OSError:
            pass
    return count


def copy_file_safely(src: Path, dst: Path, overwrite: bool = False) -> bool:
    """
    Copy a file with safety checks.

    Args:
        src: Source file path.
        dst: Destination file path.
        overwrite: Whether to overwrite if destination exists.

    Returns:
        True if successful, False otherwise.
    """
    src = Path(src)
    dst = Path(dst)

    if not src.is_file():
        return False

    if dst.exists() and not overwrite:
        return False

    try:
        ensure_parent_exists(dst)
        shutil.copy2(src, dst)
        return True
    except (OSError, shutil.Error):
        return False
