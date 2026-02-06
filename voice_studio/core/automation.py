"""
Automation system for Mira Voice Studio.

Inspired by Ableton Live's automation lanes, this module provides:
- Per-sentence parameter automation (speed, volume, pause, crossfade)
- Automation points with interpolation (linear, curved)
- Lane-based organization for visual editing
- Project-level automation storage and recall
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class InterpolationType(Enum):
    """Interpolation mode between automation points."""
    STEP = "step"        # Jump to new value (no transition)
    LINEAR = "linear"    # Linear ramp between points
    SMOOTH = "smooth"    # Ease in/out curve
    EXPONENTIAL = "exponential"  # Logarithmic curve (good for volume)


class AutomationParameter(Enum):
    """Available automation parameters."""
    SPEED = "speed"              # Speaking rate (0.5-2.0)
    VOLUME = "volume"            # Output volume (0.0-2.0)
    PAUSE_BEFORE = "pause_before"  # Pause before sentence (ms)
    PAUSE_AFTER = "pause_after"    # Pause after sentence (ms)
    CROSSFADE = "crossfade"        # Crossfade duration (ms)


# Parameter ranges and defaults
PARAMETER_CONFIG = {
    AutomationParameter.SPEED: {
        "min": 0.5,
        "max": 2.0,
        "default": 1.0,
        "unit": "x",
        "description": "Speaking speed multiplier",
    },
    AutomationParameter.VOLUME: {
        "min": 0.0,
        "max": 2.0,
        "default": 1.0,
        "unit": "",
        "description": "Volume level",
    },
    AutomationParameter.PAUSE_BEFORE: {
        "min": 0,
        "max": 2000,
        "default": 0,
        "unit": "ms",
        "description": "Pause before sentence",
    },
    AutomationParameter.PAUSE_AFTER: {
        "min": 0,
        "max": 2000,
        "default": 300,
        "unit": "ms",
        "description": "Pause after sentence",
    },
    AutomationParameter.CROSSFADE: {
        "min": 0,
        "max": 500,
        "default": 0,
        "unit": "ms",
        "description": "Crossfade with previous",
    },
}


@dataclass
class AutomationPoint:
    """
    A single automation point on a lane.

    Similar to Ableton's breakpoints - defines a value at a specific
    position with optional curve to the next point.
    """
    position: float  # Position in timeline (0.0-1.0 normalized, or sentence index)
    value: float     # Parameter value at this point
    interpolation: InterpolationType = InterpolationType.LINEAR
    curve_tension: float = 0.0  # -1.0 to 1.0, controls curve shape

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "position": self.position,
            "value": self.value,
            "interpolation": self.interpolation.value,
            "curve_tension": self.curve_tension,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutomationPoint":
        """Create from dictionary."""
        return cls(
            position=data["position"],
            value=data["value"],
            interpolation=InterpolationType(data.get("interpolation", "linear")),
            curve_tension=data.get("curve_tension", 0.0),
        )


@dataclass
class AutomationLane:
    """
    A single automation lane for one parameter.

    Like Ableton's automation lanes - contains points that define
    how a parameter changes over time.
    """
    parameter: AutomationParameter
    points: List[AutomationPoint] = field(default_factory=list)
    enabled: bool = True

    def add_point(self, position: float, value: float,
                  interpolation: InterpolationType = InterpolationType.LINEAR) -> None:
        """Add an automation point, maintaining sorted order."""
        point = AutomationPoint(position, value, interpolation)
        self.points.append(point)
        self.points.sort(key=lambda p: p.position)

    def remove_point(self, index: int) -> None:
        """Remove a point by index."""
        if 0 <= index < len(self.points):
            self.points.pop(index)

    def clear(self) -> None:
        """Remove all automation points."""
        self.points.clear()

    def get_value_at(self, position: float) -> float:
        """
        Get the interpolated value at a given position.

        Args:
            position: Position to query (0.0-1.0 or sentence index)

        Returns:
            Interpolated parameter value
        """
        if not self.points:
            return PARAMETER_CONFIG[self.parameter]["default"]

        # Before first point
        if position <= self.points[0].position:
            return self.points[0].value

        # After last point
        if position >= self.points[-1].position:
            return self.points[-1].value

        # Find surrounding points
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]

            if p1.position <= position <= p2.position:
                return self._interpolate(p1, p2, position)

        return self.points[-1].value

    def _interpolate(self, p1: AutomationPoint, p2: AutomationPoint,
                     position: float) -> float:
        """Interpolate between two points."""
        if p1.interpolation == InterpolationType.STEP:
            return p1.value

        # Normalized position between points (0-1)
        if p2.position == p1.position:
            t = 0.0
        else:
            t = (position - p1.position) / (p2.position - p1.position)

        if p1.interpolation == InterpolationType.LINEAR:
            return p1.value + (p2.value - p1.value) * t

        elif p1.interpolation == InterpolationType.SMOOTH:
            # Ease in-out using smoothstep
            t = t * t * (3 - 2 * t)
            return p1.value + (p2.value - p1.value) * t

        elif p1.interpolation == InterpolationType.EXPONENTIAL:
            # Exponential interpolation (good for volume)
            # Apply curve tension for control
            tension = p1.curve_tension
            if tension >= 0:
                t = t ** (1 + tension * 2)
            else:
                t = 1 - (1 - t) ** (1 + abs(tension) * 2)
            return p1.value + (p2.value - p1.value) * t

        return p1.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "parameter": self.parameter.value,
            "points": [p.to_dict() for p in self.points],
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutomationLane":
        """Create from dictionary."""
        return cls(
            parameter=AutomationParameter(data["parameter"]),
            points=[AutomationPoint.from_dict(p) for p in data.get("points", [])],
            enabled=data.get("enabled", True),
        )


@dataclass
class SentenceAutomation:
    """
    Per-sentence automation overrides.

    These are direct value overrides for a specific sentence,
    like clip automation in Ableton. Takes precedence over lane automation.
    """
    sentence_index: int
    speed: Optional[float] = None
    volume: Optional[float] = None
    pause_before_ms: Optional[int] = None
    pause_after_ms: Optional[int] = None
    crossfade_ms: Optional[int] = None

    def has_overrides(self) -> bool:
        """Check if any overrides are set."""
        return any([
            self.speed is not None,
            self.volume is not None,
            self.pause_before_ms is not None,
            self.pause_after_ms is not None,
            self.crossfade_ms is not None,
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sentence_index": self.sentence_index,
            "speed": self.speed,
            "volume": self.volume,
            "pause_before_ms": self.pause_before_ms,
            "pause_after_ms": self.pause_after_ms,
            "crossfade_ms": self.crossfade_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceAutomation":
        """Create from dictionary."""
        return cls(
            sentence_index=data["sentence_index"],
            speed=data.get("speed"),
            volume=data.get("volume"),
            pause_before_ms=data.get("pause_before_ms"),
            pause_after_ms=data.get("pause_after_ms"),
            crossfade_ms=data.get("crossfade_ms"),
        )


@dataclass
class AutomationProject:
    """
    Complete automation state for a project.

    Contains both lane-based automation (global curves) and
    per-sentence overrides (clip automation).
    """
    lanes: Dict[AutomationParameter, AutomationLane] = field(default_factory=dict)
    sentence_overrides: Dict[int, SentenceAutomation] = field(default_factory=dict)

    # Global defaults (can be overridden)
    default_speed: float = 1.0
    default_volume: float = 1.0
    default_pause_ms: int = 300

    def __post_init__(self):
        """Initialize default lanes if not provided."""
        for param in AutomationParameter:
            if param not in self.lanes:
                self.lanes[param] = AutomationLane(parameter=param)

    def get_sentence_params(self, sentence_index: int,
                           total_sentences: int) -> Dict[str, Any]:
        """
        Get all automation parameters for a sentence.

        Priority: Sentence override > Lane automation > Global default

        Args:
            sentence_index: 1-based sentence index
            total_sentences: Total number of sentences (for normalization)

        Returns:
            Dictionary of parameter values
        """
        # Normalize position (0.0 to 1.0)
        if total_sentences > 1:
            normalized_pos = (sentence_index - 1) / (total_sentences - 1)
        else:
            normalized_pos = 0.0

        # Start with defaults
        params = {
            "speed": self.default_speed,
            "volume": self.default_volume,
            "pause_before_ms": 0,
            "pause_after_ms": self.default_pause_ms,
            "crossfade_ms": 0,
        }

        # Apply lane automation
        param_map = {
            AutomationParameter.SPEED: "speed",
            AutomationParameter.VOLUME: "volume",
            AutomationParameter.PAUSE_BEFORE: "pause_before_ms",
            AutomationParameter.PAUSE_AFTER: "pause_after_ms",
            AutomationParameter.CROSSFADE: "crossfade_ms",
        }

        for auto_param, key in param_map.items():
            lane = self.lanes.get(auto_param)
            if lane and lane.enabled and lane.points:
                params[key] = lane.get_value_at(normalized_pos)

        # Apply sentence overrides (highest priority)
        override = self.sentence_overrides.get(sentence_index)
        if override:
            if override.speed is not None:
                params["speed"] = override.speed
            if override.volume is not None:
                params["volume"] = override.volume
            if override.pause_before_ms is not None:
                params["pause_before_ms"] = override.pause_before_ms
            if override.pause_after_ms is not None:
                params["pause_after_ms"] = override.pause_after_ms
            if override.crossfade_ms is not None:
                params["crossfade_ms"] = override.crossfade_ms

        return params

    def set_sentence_override(self, sentence_index: int, **kwargs) -> None:
        """Set or update a sentence override."""
        if sentence_index not in self.sentence_overrides:
            self.sentence_overrides[sentence_index] = SentenceAutomation(
                sentence_index=sentence_index
            )

        override = self.sentence_overrides[sentence_index]
        for key, value in kwargs.items():
            if hasattr(override, key):
                setattr(override, key, value)

    def clear_sentence_override(self, sentence_index: int) -> None:
        """Remove all overrides for a sentence."""
        if sentence_index in self.sentence_overrides:
            del self.sentence_overrides[sentence_index]

    def add_lane_point(self, parameter: AutomationParameter,
                       position: float, value: float,
                       interpolation: InterpolationType = InterpolationType.LINEAR) -> None:
        """Add a point to an automation lane."""
        if parameter not in self.lanes:
            self.lanes[parameter] = AutomationLane(parameter=parameter)
        self.lanes[parameter].add_point(position, value, interpolation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "lanes": {k.value: v.to_dict() for k, v in self.lanes.items()},
            "sentence_overrides": {
                str(k): v.to_dict() for k, v in self.sentence_overrides.items()
            },
            "default_speed": self.default_speed,
            "default_volume": self.default_volume,
            "default_pause_ms": self.default_pause_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutomationProject":
        """Create from dictionary."""
        lanes = {}
        for k, v in data.get("lanes", {}).items():
            param = AutomationParameter(k)
            lanes[param] = AutomationLane.from_dict(v)

        overrides = {}
        for k, v in data.get("sentence_overrides", {}).items():
            overrides[int(k)] = SentenceAutomation.from_dict(v)

        return cls(
            lanes=lanes,
            sentence_overrides=overrides,
            default_speed=data.get("default_speed", 1.0),
            default_volume=data.get("default_volume", 1.0),
            default_pause_ms=data.get("default_pause_ms", 300),
        )

    def save(self, path: Path) -> None:
        """Save automation project to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "AutomationProject":
        """Load automation project from file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# Convenience functions for common automation patterns

def create_fade_in(lane: AutomationLane, duration: float = 0.1) -> None:
    """Add a fade-in automation (volume 0 to 1)."""
    lane.clear()
    lane.add_point(0.0, 0.0, InterpolationType.SMOOTH)
    lane.add_point(duration, 1.0, InterpolationType.LINEAR)


def create_fade_out(lane: AutomationLane, start: float = 0.9) -> None:
    """Add a fade-out automation (volume 1 to 0)."""
    lane.add_point(start, 1.0, InterpolationType.LINEAR)
    lane.add_point(1.0, 0.0, InterpolationType.SMOOTH)


def create_speed_ramp(lane: AutomationLane,
                      start_speed: float = 1.0,
                      end_speed: float = 1.2) -> None:
    """Add a gradual speed ramp."""
    lane.clear()
    lane.add_point(0.0, start_speed, InterpolationType.LINEAR)
    lane.add_point(1.0, end_speed, InterpolationType.LINEAR)


def create_dramatic_pause(project: AutomationProject,
                          sentence_index: int,
                          pause_ms: int = 1000) -> None:
    """Add a dramatic pause before a specific sentence."""
    project.set_sentence_override(
        sentence_index,
        pause_before_ms=pause_ms,
        speed=0.9  # Slightly slower for emphasis
    )


def apply_volume_envelope(audio: np.ndarray,
                          sample_rate: int,
                          lane: AutomationLane) -> np.ndarray:
    """
    Apply volume automation to an audio array.

    Args:
        audio: Audio samples
        sample_rate: Sample rate in Hz
        lane: Volume automation lane

    Returns:
        Audio with volume automation applied
    """
    if not lane.points or not lane.enabled:
        return audio

    duration = len(audio) / sample_rate

    # Create envelope array
    envelope = np.zeros(len(audio))

    for i in range(len(audio)):
        position = i / len(audio)  # Normalized position
        envelope[i] = lane.get_value_at(position)

    return audio * envelope
