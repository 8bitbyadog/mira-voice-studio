"""
Training script generator for Mira Voice Studio.

Generates optimal scripts for users to read aloud during voice training.
Scripts are designed to:
1. Cover all phonemes in the target language
2. Capture emotional range
3. Include natural speech patterns
4. Vary sentence length
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import random


@dataclass
class ScriptLine:
    """A single line from a training script."""

    index: int
    text: str
    category: str  # "phoneme", "emotional", "conversational", "custom"
    emotion: Optional[str] = None  # For emotional scripts
    notes: Optional[str] = None


# Phoneme coverage sentences (English)
PHONEME_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck?",
    "Peter Piper picked a peck of pickled peppers.",
    "The sixth sick sheik's sixth sheep's sick.",
    "Red lorry, yellow lorry, red lorry, yellow lorry.",
    "Unique New York, you know you need unique New York.",
    "A proper copper coffee pot.",
    "Around the rugged rocks the ragged rascal ran.",
    "Betty Botter bought some butter.",
    "I saw Susie sitting in a shoeshine shop.",
    "Fresh French fried fish.",
    "Toy boat, toy boat, toy boat.",
    "Lesser leather never weathered wetter weather better.",
    "Six thick thistle sticks.",
    "Which wristwatches are Swiss wristwatches?",
    "The thirty-three thieves thought that they thrilled the throne.",
    "Can you can a can as a canner can can a can?",
    "I scream, you scream, we all scream for ice cream.",
    "Fuzzy Wuzzy was a bear, Fuzzy Wuzzy had no hair.",
    "A big black bug bit a big black bear.",
    "Eleven benevolent elephants.",
    "Greek grapes grow in Greece.",
    "The great Greek grape growers grow great Greek grapes.",
    "I wish to wash my Irish wristwatch.",
    "Near an ear, a nearer ear, a nearly eerie ear.",
    "Six sleek swans swam swiftly southwards.",
    "Black background, brown background.",
    "Pad kid poured curd pulled cod.",
    "A skunk sat on a stump and thunk the stump stunk.",
]

# Emotional range sentences
EMOTIONAL_SENTENCES = {
    "neutral": [
        "The meeting is scheduled for three o'clock tomorrow.",
        "Please turn left at the next intersection.",
        "The temperature today will reach seventy-five degrees.",
        "Your order has been processed and will ship tomorrow.",
        "The document contains all the information you requested.",
        "Remember to save your work before closing the application.",
    ],
    "happy": [
        "I can't believe we actually won the championship!",
        "This is the best news I've heard all year!",
        "I'm so excited to finally meet you in person!",
        "We did it! We actually did it!",
        "This is absolutely wonderful, thank you so much!",
        "I've been looking forward to this moment for so long!",
    ],
    "serious": [
        "This is a matter that requires your immediate attention.",
        "We need to discuss something important.",
        "I want to be very clear about what I'm about to say.",
        "Please listen carefully to the following instructions.",
        "This decision will have significant consequences.",
        "I need you to understand the gravity of this situation.",
    ],
    "excited": [
        "Oh my god, you're never going to believe what just happened!",
        "Guess what? I got the job! I actually got the job!",
        "This is incredible, absolutely incredible!",
        "I've never seen anything like this before!",
        "Wait until you hear this, it's amazing!",
        "You won't believe what I just found out!",
    ],
    "warm": [
        "Hey, it's really good to see you again.",
        "I just wanted to check in and see how you're doing.",
        "Take care of yourself, okay?",
        "I'm here for you if you need anything.",
        "Thank you for being such a good friend.",
        "I really appreciate everything you've done.",
    ],
    "curious": [
        "Wait, what exactly do you mean by that?",
        "Hmm, that's interesting. Can you tell me more?",
        "I wonder what would happen if we tried something different.",
        "Have you ever thought about why things work that way?",
        "So how does that actually work?",
        "What made you decide to go in that direction?",
    ],
    "thoughtful": [
        "Let me think about that for a moment.",
        "That's a really good question, actually.",
        "I've been giving this a lot of thought lately.",
        "There are several ways we could approach this.",
        "Consider the implications of what you're suggesting.",
        "It's worth taking the time to examine this carefully.",
    ],
}

# Conversational YouTube-style sentences
CONVERSATIONAL_SENTENCES = [
    "Hey everyone, welcome back to the channel.",
    "So today I want to talk about something really interesting.",
    "Before we dive in, make sure to hit that subscribe button.",
    "Alright, let's get into it.",
    "Now, here's where things get interesting.",
    "And that's basically the main idea.",
    "Let me know what you think in the comments below.",
    "Thanks for watching, and I'll see you in the next one!",
    "What's up guys, hope you're having a great day.",
    "So I've been working on something really cool.",
    "If you're new here, welcome to the channel.",
    "Okay, so here's the thing.",
    "I'm going to show you exactly how to do this.",
    "First things first, let's talk about the basics.",
    "Now this next part is really important.",
    "Pay close attention to this part.",
    "Alright, moving on to the next topic.",
    "This is something I get asked about a lot.",
    "And there you have it, that's all there is to it.",
    "Don't forget to leave a like if you found this helpful.",
    "I'll put links to everything in the description below.",
    "Let's break this down step by step.",
    "Here's a quick recap of what we covered.",
    "Stay tuned because we've got more coming up.",
    "That wraps up today's video.",
]

# Questions for natural speech patterns
QUESTION_SENTENCES = [
    "What do you think about this approach?",
    "Have you ever experienced something like that?",
    "Does this make sense so far?",
    "Can you see where I'm going with this?",
    "Isn't that just fascinating?",
    "Would you be interested in learning more?",
    "How does that sound to you?",
    "What would you do in this situation?",
    "Do you have any questions about this?",
    "Ready to move on to the next step?",
]

# Exclamations for dynamic range
EXCLAMATION_SENTENCES = [
    "Wow, that's amazing!",
    "Oh no, that's not good!",
    "Wait, really?",
    "No way!",
    "That's perfect!",
    "Incredible!",
    "Oh, I see what you mean now!",
    "Exactly!",
    "Absolutely!",
    "Of course!",
]


class ScriptGenerator:
    """
    Generate training scripts optimized for voice model training.

    Script types:
    - phoneme: Covers all English phonemes
    - emotional: Various emotional expressions
    - conversational: Natural YouTube/podcast style
    - mixed: Combination of all types
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the script generator.

        Args:
            language: Target language (currently only "en" supported).
        """
        self.language = language

    def generate_phoneme_script(
        self,
        num_sentences: int = 30
    ) -> List[ScriptLine]:
        """Generate script for phoneme coverage."""
        sentences = PHONEME_SENTENCES.copy()
        random.shuffle(sentences)

        if num_sentences < len(sentences):
            sentences = sentences[:num_sentences]

        return [
            ScriptLine(
                index=i + 1,
                text=text,
                category="phoneme",
                notes="Read naturally, as if talking to a friend.",
            )
            for i, text in enumerate(sentences)
        ]

    def generate_emotional_script(
        self,
        num_sentences: int = 30,
        emotions: Optional[List[str]] = None
    ) -> List[ScriptLine]:
        """Generate script for emotional range."""
        if emotions is None:
            emotions = list(EMOTIONAL_SENTENCES.keys())

        lines = []
        index = 1

        # Distribute sentences across emotions
        sentences_per_emotion = max(1, num_sentences // len(emotions))

        for emotion in emotions:
            emotion_sentences = EMOTIONAL_SENTENCES.get(emotion, [])
            selected = emotion_sentences[:sentences_per_emotion]

            for text in selected:
                lines.append(ScriptLine(
                    index=index,
                    text=text,
                    category="emotional",
                    emotion=emotion,
                    notes=f"Read with {emotion} emotion.",
                ))
                index += 1

                if index > num_sentences:
                    break

            if index > num_sentences:
                break

        return lines

    def generate_conversational_script(
        self,
        num_sentences: int = 25
    ) -> List[ScriptLine]:
        """Generate conversational YouTube-style script."""
        sentences = (
            CONVERSATIONAL_SENTENCES +
            QUESTION_SENTENCES +
            EXCLAMATION_SENTENCES
        )
        random.shuffle(sentences)

        if num_sentences < len(sentences):
            sentences = sentences[:num_sentences]

        return [
            ScriptLine(
                index=i + 1,
                text=text,
                category="conversational",
                notes="Read as if recording a video for your channel.",
            )
            for i, text in enumerate(sentences)
        ]

    def generate_mixed_script(
        self,
        num_sentences: int = 50
    ) -> List[ScriptLine]:
        """Generate a mixed script with variety."""
        # Allocate sentences to each type
        phoneme_count = num_sentences // 3
        emotional_count = num_sentences // 3
        conversational_count = num_sentences - phoneme_count - emotional_count

        phoneme = self.generate_phoneme_script(phoneme_count)
        emotional = self.generate_emotional_script(emotional_count)
        conversational = self.generate_conversational_script(conversational_count)

        # Combine and shuffle
        all_lines = phoneme + emotional + conversational
        random.shuffle(all_lines)

        # Re-index
        for i, line in enumerate(all_lines):
            line.index = i + 1

        return all_lines

    def generate_custom_script(
        self,
        text: str
    ) -> List[ScriptLine]:
        """
        Generate script from custom text input.

        Args:
            text: Custom text to use as script.

        Returns:
            List of ScriptLine objects.
        """
        from voice_studio.core.text_processor import TextProcessor

        processor = TextProcessor()
        sentences = processor.split_sentences(text)

        return [
            ScriptLine(
                index=i + 1,
                text=sentence,
                category="custom",
            )
            for i, sentence in enumerate(sentences)
        ]

    def generate(
        self,
        script_type: str = "mixed",
        num_sentences: int = 50,
        **kwargs
    ) -> List[ScriptLine]:
        """
        Generate a training script.

        Args:
            script_type: Type of script ("phoneme", "emotional", "conversational", "mixed").
            num_sentences: Target number of sentences.
            **kwargs: Additional arguments for specific script types.

        Returns:
            List of ScriptLine objects.
        """
        generators = {
            "phoneme": self.generate_phoneme_script,
            "emotional": self.generate_emotional_script,
            "conversational": self.generate_conversational_script,
            "mixed": self.generate_mixed_script,
        }

        generator = generators.get(script_type, self.generate_mixed_script)
        return generator(num_sentences, **kwargs) if kwargs else generator(num_sentences)

    def to_text(self, lines: List[ScriptLine], include_notes: bool = False) -> str:
        """
        Convert script lines to plain text.

        Args:
            lines: List of ScriptLine objects.
            include_notes: Whether to include notes/emotions.

        Returns:
            Formatted text string.
        """
        output = []

        for line in lines:
            if include_notes and (line.emotion or line.notes):
                prefix = f"[{line.emotion.upper()}] " if line.emotion else ""
                output.append(f"{line.index}. {prefix}{line.text}")
                if line.notes:
                    output.append(f"   Note: {line.notes}")
            else:
                output.append(f"{line.index}. {line.text}")

        return "\n".join(output)

    def save_script(
        self,
        lines: List[ScriptLine],
        output_path: Path,
        include_notes: bool = True
    ) -> None:
        """Save script to a text file."""
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text = self.to_text(lines, include_notes=include_notes)
        output_path.write_text(text, encoding="utf-8")
