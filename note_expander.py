import argparse
import concurrent.futures
import csv
import multiprocessing
import os
import re
import shutil
import signal
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import librosa.effects
import numpy as np
import psutil
import questionary
import sounddevice as sd
import soundfile as sf
from colorama import Fore, Style, init
from scipy import signal as scipy_signal
from tqdm import tqdm

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Constants for colored output
SUCCESS = Fore.GREEN
ERROR = Fore.RED
WARNING = Fore.YELLOW
INFO = Fore.CYAN
HIGHLIGHT = Fore.MAGENTA
RESET = Style.RESET_ALL

# Mapping of original chord names to filename-safe shorthand
CHORD_SHORTHAND = {
    "Major fifth": "Major5",
    "Dominant seventh": "Dom7",
    "Major seventh": "Maj7",
    "Major sixth": "Maj6",
    "Dominant Minor ninth": "Dom7b9",
    "Dominant ninth": "Dom9",
    "Dominant seventh sharp ninth": "Dom7sharp9",
    "Lydian": "Lydian",
    "Major sixth ninth": "Maj6_9",
    "Major ninth": "Maj9",
    "Seven six": "Seven6",
    "Augmented eleventh": "Aug11",
    "Dominant eleventh": "Dom11",
    "Major eleventh": "Maj11",
    "Thirteenth flat ninth": "Dom13b9",
    "Dominant thirteenth": "Dom13",
    "Major thirteenth": "Maj13",
    "Minor fifth": "Minor5",
    "Minor Major seventh": "MinMaj7",
    "Minor seventh": "Min7",
    "Minor sixth": "Min6",
    "Minor ninth": "Min9",
    "Minor sixth ninth": "Min6_9",
    "Minor eleventh": "Min11",
    "Minor thirteenth": "Min13",
    "Augmented": "Aug",
    "Augmented Major seventh": "AugMaj7",
    "Augmented seventh": "Aug7",
    "Major seventh sharp eleventh": "Maj7sharp11",
    "Ninth Augmented fifth": "Dom9Aug5",
    "Diminished": "Dim",
    "Diminished Major seventh": "DimMaj7",
    "Diminished seventh": "Dim7",
    "Half-Diminished seventh": "HalfDim7",
    "Power chord": "PowerChord",
    "Augmented sixth Italian": "Italian6",
    "Augmented sixth French": "French6",
    "Augmented sixth German": "German6",
    "Tristan chord": "Tristan",
    "Suspended": "Sus4",
    "Seventh suspension four": "7Sus4",
    "Ninth flat fifth": "Dom9b5",
    "Thirteenth flat ninth flat fifth": "Dom13b9b5",
    "Dream chord": "Dream",
    "Magic chord": "Magic",
    "Elektra": "Elektra",
    "So What": "SoWhat",
    "Petrushka": "Petrushka",
    "Farben chord": "Farben",
    "Viennese trichord two forms": "VienneseTri",
    "Mystic chord": "Mystic",
    "Ode-to-Napoleon hexachord": "OdeNapoleon",
    "Northern lights": "NorthernLights",
}


def chord_to_filename(chord_name):
    """Convert a chord name to a shortened filename-friendly version"""
    return CHORD_SHORTHAND.get(chord_name, chord_name.replace(" ", ""))


# Define questionary custom style
custom_style = questionary.Style(
    [
        ("qmark", "fg:cyan bold"),  # question mark
        ("question", "fg:white bold"),  # question text
        ("answer", "fg:green bold"),  # submitted answer
        ("pointer", "fg:cyan bold"),  # pointer used in select and checkbox prompts
        (
            "highlighted",
            "fg:cyan bold",
        ),  # highlighted choice in select and checkbox prompts
        ("selected", "fg:green bold"),  # selected choice in checkbox prompts
        ("separator", "fg:cyan"),  # separator in lists
        ("instruction", "fg:white"),  # user instructions for select, checkbox prompts
        ("text", "fg:white"),  # plain text
        (
            "disabled",
            "fg:gray italic",
        ),  # disabled choices for select and checkbox prompts
    ]
)

# Global lock for tqdm progress bars to avoid display issues in multi-threading
tqdm_lock = threading.Lock()

# Global status dictionary to track progress of each directory
processing_status = {}
status_lock = threading.Lock()


def update_status(directory, message, status_type="info"):
    """Update the status of a directory processing task."""
    with status_lock:
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        processing_status[directory] = {
            "message": message,
            "timestamp": timestamp,
            "type": status_type,
        }

        # Print the update with appropriate color
        if status_type == "info":
            print(
                f"{INFO}[{timestamp}] {os.path.basename(directory)}: {message}{RESET}"
            )
        elif status_type == "success":
            print(
                f"{SUCCESS}[{timestamp}] {os.path.basename(directory)}: {message}{RESET}"
            )
        elif status_type == "warning":
            print(
                f"{WARNING}[{timestamp}] {os.path.basename(directory)}: {message}{RESET}"
            )
        elif status_type == "error":
            print(
                f"{ERROR}[{timestamp}] {os.path.basename(directory)}: {message}{RESET}"
            )


def get_optimal_workers():
    """Determine the optimal number of worker threads based on system resources.

    Returns a reasonable number of workers based on CPU cores and available memory,
    with safeguards to prevent system overload.
    """
    # Get system information
    cpu_count = multiprocessing.cpu_count()
    memory = psutil.virtual_memory()

    # Base number of workers on CPU cores
    # Use N-1 cores on systems with more than 4 cores to leave one core for the OS
    if cpu_count > 4:
        workers = cpu_count - 1
    else:
        workers = max(1, cpu_count // 2)

    # Adjust based on available memory
    # Audio processing can be memory-intensive, so limit workers if memory is low
    memory_gb = memory.available / (1024 * 1024 * 1024)

    # Estimate ~2GB per worker for audio processing
    memory_limited_workers = max(1, int(memory_gb / 2))

    # Take the smaller of the two calculations
    workers = min(workers, memory_limited_workers)

    # Never use more than 8 workers to prevent system overload
    workers = min(workers, 8)

    return max(1, workers)  # Always return at least 1 worker


def print_header(text):
    """Print a formatted header."""
    print(f"\n{INFO}{'='*60}{RESET}")
    print(f"{INFO}{text}{RESET}")
    print(f"{INFO}{'='*60}{RESET}")


def print_success(text):
    """Print a success message."""
    print(f"{SUCCESS}{text}{RESET}")


def print_warning(text):
    """Print a warning message."""
    print(f"{WARNING}{text}{RESET}")


def print_error(text):
    """Print an error message."""
    print(f"{ERROR}{text}{RESET}")


def print_info(text):
    """Print an info message."""
    print(f"{INFO}{text}{RESET}")


def print_highlight(text):
    """Print a highlighted message."""
    print(f"{HIGHLIGHT}{text}{RESET}")


def parse_note_from_filename(filename):
    """Extract note and octave from filename."""
    # First try to match patterns with sharps like 'CC Piano-G#5.wav' or 'CC Piano-G3.wav'
    match = re.search(r"([A-G]#?)(\d+)", filename)
    if match:
        note, octave = match.groups()
        return note, int(octave)

    # If no match, try to match patterns with flats like 'CC Piano-Bb4.wav' or 'CC Piano-Eb2.wav'
    match = re.search(r"([A-G]b)(\d+)", filename)
    if match:
        flat_note, octave = match.groups()
        # Convert flat note to equivalent sharp note
        flat_to_sharp = {
            "Cb": "B",  # B is in the previous octave, but we'll handle that separately
            "Db": "C#",
            "Eb": "D#",
            "Fb": "E",  # E has no sharp equivalent
            "Gb": "F#",
            "Ab": "G#",
            "Bb": "A#",
        }

        # Special case for Cb which is actually B in the previous octave
        if flat_note == "Cb":
            return "B", int(octave) - 1

        # Return the equivalent sharp note
        return flat_to_sharp[flat_note], int(octave)

    return None, None


def get_note_frequency(note, octave):
    """
    Calculate the frequency of a note based on its name and octave.
    A4 (440 Hz) is used as the reference.
    """
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Handle flat notes by converting to their sharp equivalents
    if note.endswith("b"):
        flat_to_sharp = {
            "Cb": "B",
            "Db": "C#",
            "Eb": "D#",
            "Fb": "E",
            "Gb": "F#",
            "Ab": "G#",
            "Bb": "A#",
        }
        if note in flat_to_sharp:
            note = flat_to_sharp[note]
            if note == "B":  # Special case: Cb is B in the previous octave
                octave -= 1

    # A4 is the reference note at 440 Hz
    A4_FREQ = 440.0
    A4_NOTE_INDEX = notes.index("A")
    A4_OCTAVE = 4

    # Calculate the number of semitones from A4
    note_index = notes.index(note)
    semitones_from_a4 = (octave - A4_OCTAVE) * 12 + (note_index - A4_NOTE_INDEX)

    # Calculate the frequency using the equal temperament formula
    # f = f_ref * 2^(n/12) where n is the number of semitones from the reference note
    frequency = A4_FREQ * (2 ** (semitones_from_a4 / 12))

    return frequency


def get_all_wav_files(directory="."):
    """Get all WAV files in the specified directory."""
    # Exclude files with -00-Full in the filename
    wav_files = [
        f
        for f in os.listdir(directory)
        if f.lower().endswith(".wav") and "-00-Full" not in f
    ]

    # Debug: Print the notes found in each file
    if wav_files:
        print_info(f"Found {len(wav_files)} WAV files in {directory}")
        for wav_file in wav_files:
            note, octave = parse_note_from_filename(wav_file)
            if note and octave:
                print_info(f"  Detected {note}{octave} in file: {wav_file}")
            else:
                print_warning(f"  Could not detect note in file: {wav_file}")

    return wav_files


def pitch_shift_sample(
    audio_data, sr, source_note, source_octave, target_note, target_octave
):
    """Pitch shift a sample from source note to target note."""
    source_freq = get_note_frequency(source_note, source_octave)
    target_freq = get_note_frequency(target_note, target_octave)

    # Calculate pitch shift factor
    shift_factor = target_freq / source_freq

    # Resample the audio
    if shift_factor != 1.0:
        # For higher notes, we need to shorten the sample (speed up)
        # For lower notes, we need to lengthen the sample (slow down)
        new_length = int(len(audio_data) / shift_factor)
        # Ensure we return a numpy array, not a tuple
        resampled_audio = scipy_signal.resample(audio_data, new_length)
        return resampled_audio, sr

    return audio_data, sr


def find_closest_sample(target_note, target_octave, existing_samples):
    """Find the closest existing sample to use as a source."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Debug information
    print_info(f"Finding closest sample for {target_note}{target_octave}")
    print_info(f"Available samples: {len(existing_samples)}")

    # First, check if we're looking for an extreme octave (very high or very low)
    # If so, we'll limit our search to the available octaves
    available_octaves = set()
    for sample in existing_samples:
        note, octave = parse_note_from_filename(sample)
        if note and octave:
            available_octaves.add(octave)

    print_info(f"Available octaves: {sorted(available_octaves)}")

    # If target octave is outside available range, adjust it to the closest available
    if target_octave not in available_octaves and available_octaves:
        original_octave = target_octave
        if target_octave > max(available_octaves):
            target_octave = max(available_octaves)
        elif target_octave < min(available_octaves):
            target_octave = min(available_octaves)
        print_info(
            f"Adjusted target octave from {original_octave} to {target_octave} (available range)"
        )

    target_index = notes.index(target_note) + (target_octave * 12)

    closest_sample = None
    min_distance = float("inf")
    exact_match = None

    for sample in existing_samples:
        note, octave = parse_note_from_filename(sample)
        if note and octave:
            # Check for exact match first (accounting for enharmonic equivalents)
            if octave == target_octave:
                if note == target_note:
                    # Direct match
                    exact_match = sample
                    print_info(f"Found exact match: {sample}")
                    break

            # Calculate semitone distance
            sample_index = notes.index(note) + (octave * 12)
            distance = abs(target_index - sample_index)

            if distance < min_distance:
                min_distance = distance
                closest_sample = sample
                print_info(
                    f"New closest sample: {sample} (distance: {distance} semitones)"
                )

    # Return exact match if found, otherwise closest sample
    result = exact_match if exact_match else closest_sample
    print_info(f"Selected sample: {result}")
    return result


# Update chord definitions to match Chords.csv format exactly
CHORD_DEFINITIONS = [
    # Format: (Chord name, Quality, Semitones list, Notes count)
    ("Major fifth", "Major", [0, 4, 7], 3),
    ("Dominant seventh", "Major", [0, 4, 7, 10], 4),
    ("Major seventh", "Major", [0, 4, 7, 11], 4),
    ("Major sixth", "Major", [0, 4, 7, 9], 4),
    ("Dominant Minor ninth", "Major", [0, 4, 7, 10, 13], 5),
    ("Dominant ninth", "Major", [0, 4, 7, 10, 14], 5),
    ("Dominant seventh sharp ninth", "Major", [0, 4, 7, 10, 15], 5),
    ("Lydian", "Major", [0, 4, 7, 11, 18], 5),
    ("Major sixth ninth", "Major", [0, 4, 7, 9, 14], 5),
    ("Major ninth", "Major", [0, 4, 7, 11, 14], 5),
    ("Seven six", "Major", [0, 4, 7, 9, 10], 5),
    ("Augmented eleventh", "Major", [0, 4, 7, 10, 14, 18], 6),
    ("Dominant eleventh", "Major", [0, 4, 7, 10, 14, 17], 6),
    ("Major eleventh", "Major", [0, 4, 7, 11, 14, 17], 6),
    ("Thirteenth flat ninth", "Major", [0, 4, 7, 10, 13, 21], 6),
    ("Dominant thirteenth", "Major", [0, 4, 7, 10, 14, 17, 21], 7),
    ("Major thirteenth", "Major", [0, 4, 7, 11, 14, 17, 21], 7),
    ("Minor fifth", "Minor", [0, 3, 7], 3),
    ("Minor Major seventh", "Minor", [0, 3, 7, 11], 4),
    ("Minor seventh", "Minor", [0, 3, 7, 10], 4),
    ("Minor sixth", "Minor", [0, 3, 7, 9], 4),
    ("Minor ninth", "Minor", [0, 3, 7, 10, 14], 5),
    ("Minor sixth ninth", "Minor", [0, 3, 7, 9, 14], 5),
    ("Minor eleventh", "Minor", [0, 3, 7, 10, 14, 17], 6),
    ("Minor thirteenth", "Minor", [0, 3, 7, 10, 14, 17, 21], 7),
    ("Augmented", "Augmented", [0, 4, 8], 3),
    ("Augmented Major seventh", "Augmented", [0, 4, 8, 11], 4),
    ("Augmented seventh", "Augmented", [0, 4, 8, 10], 4),
    ("Major seventh sharp eleventh", "Augmented", [0, 4, 8, 11, 18], 5),
    ("Ninth Augmented fifth", "Augmented", [0, 4, 8, 10, 14], 5),
    ("Diminished", "Diminished", [0, 3, 6], 3),
    ("Diminished Major seventh", "Diminished", [0, 3, 6, 11], 4),
    ("Diminished seventh", "Diminished", [0, 3, 6, 9], 4),
    ("Half-Diminished seventh", "Diminished", [0, 4, 6, 10], 4),
    ("Power chord", "Indeterminate", [0, 7], 2),
    ("Augmented sixth Italian", "Predominant", [0, 4, 10], 3),
    ("Augmented sixth French", "Predominant", [0, 4, 6, 10], 4),
    ("Augmented sixth German", "Predominant", [0, 4, 7, 10], 4),
    ("Tristan chord", "Predominant", [0, 3, 6, 10], 4),
    ("Suspended", "Suspended", [0, 5, 7], 3),
    ("Seventh suspension four", "Suspended", [0, 5, 7, 10], 4),
    ("Ninth flat fifth", "M3+d5", [0, 4, 6, 10, 14], 5),
    ("Thirteenth flat ninth flat fifth", "M3+d5", [0, 4, 6, 10, 13, 21], 6),
    ("Dream chord", "Just", [0, 5, 6, 7], 4),
    ("Magic chord", "Just", [0, 1, 5, 6, 10, 12, 15, 17], 8),
    ("Elektra", "Bitonal", [0, 7, 9, 13, 16], 5),
    ("So What", "Bitonal", [0, 5, 10, 15, 19], 5),
    ("Petrushka", "Bitonal", [0, 1, 4, 6, 7, 10], 6),
    ("Farben chord", "Atonal", [0, 8, 11, 16, 21], 5),
    ("Viennese trichord two forms", "Atonal", [0, 1, 6], 3),
    ("Mystic chord", "Atonal", [0, 6, 10, 16, 21, 26], 6),
    ("Ode-to-Napoleon hexachord", "Atonal", [0, 1, 4, 5, 8, 9], 6),
    ("Northern lights", "Atonal", [1, 2, 8, 12, 15, 18, 19, 22, 23, 28, 31], 11),
]


def get_note_from_semitone(root_note, root_octave, semitone_offset):
    """Get the note and octave given a root note and semitone offset."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Calculate the absolute semitone position of the root note
    root_index = notes.index(root_note)
    absolute_semitone = root_octave * 12 + root_index

    # Add the offset to get the target semitone
    target_semitone = absolute_semitone + semitone_offset

    # Calculate the target octave and note
    target_octave = target_semitone // 12
    target_note_index = target_semitone % 12
    target_note = notes[target_note_index]

    return target_note, target_octave


def generate_chord_inversions(semitones):
    """Generate all possible inversions for a chord based on its semitones.

    Args:
        semitones: List of semitone offsets from the root note

    Returns:
        List of tuples, each containing (inversion_number, inverted_semitones)
        where inversion_number is 1 for first inversion, 2 for second, etc.
    """
    inversions = []
    chord_size = len(semitones)

    # Skip if chord has less than 3 notes (power chords, etc.)
    if chord_size < 3:
        return inversions

    # Generate each inversion
    for i in range(1, chord_size):
        # Create the inverted semitones list
        inverted = []

        # Move the first i notes up an octave
        for j in range(chord_size):
            if j < i:
                # Move note up an octave (add 12 semitones) and adjust relative to new root
                inverted.append(semitones[j] + 12 - semitones[i])
            else:
                # Adjust remaining notes relative to new root
                inverted.append(semitones[j] - semitones[i])

        # Sort the inverted semitones
        inverted.sort()

        inversions.append((i, inverted))

    return inversions


def generate_chord(
    root_note,
    root_octave,
    semitones,
    all_samples,
    source_dir,
    target_dir,
    prefix,
    chord_duration_factor=4.0,
):
    """Generate a chord sample by mixing multiple note samples."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Print information about the chord we're generating
    print_info(
        f"Generating chord with root {root_note}{root_octave} and semitones {semitones}"
    )

    # Check if we're dealing with extreme octaves (1, 5-8)
    is_extreme_octave = root_octave == 1 or root_octave >= 5

    # For extreme octaves, we'll use a more reliable approach
    # We'll generate the chord in a middle octave (3) and then pitch shift it
    if is_extreme_octave:
        print_info(f"Using reliable approach for extreme octave {root_octave}")

        # Generate the chord in octave 3
        middle_octave = 3
        middle_chord_audio, middle_sr = generate_chord(
            root_note,
            middle_octave,
            semitones,
            all_samples,
            source_dir,
            target_dir,
            prefix,
            chord_duration_factor,
        )

        if middle_chord_audio is None:
            print_error(f"Failed to generate middle octave chord as a base")
            return None, None

        # Calculate the pitch shift factor
        source_freq = get_note_frequency(root_note, middle_octave)
        target_freq = get_note_frequency(root_note, root_octave)
        shift_factor = target_freq / source_freq

        print_info(
            f"Pitch shifting from {root_note}{middle_octave} to {root_note}{root_octave} (factor: {shift_factor:.2f})"
        )

        # Resample the audio
        if shift_factor != 1.0:
            # For higher notes, we need to shorten the sample (speed up)
            # For lower notes, we need to lengthen the sample (slow down)
            new_length = int(len(middle_chord_audio) / shift_factor)
            new_audio = scipy_signal.resample(middle_chord_audio, new_length)

            # Time stretching to maintain consistent duration
            current_duration = len(new_audio) / middle_sr
            source_duration = len(middle_chord_audio) / middle_sr
            stretch_factor = source_duration / current_duration

            # Only apply time stretching if the difference is significant
            if abs(stretch_factor - 1.0) > 0.01:
                print_info(
                    f"Time stretching to match duration (factor: {stretch_factor:.2f})"
                )

                # For librosa's time_stretch, rate < 1 makes it longer
                rate = 1.0 / stretch_factor

                # For very short samples, use a smaller n_fft value
                n_fft = 2048  # Default value
                if len(new_audio) < n_fft:
                    n_fft = 2 ** int(np.log2(len(new_audio) - 1))
                    n_fft = max(32, n_fft)  # Ensure it's not too small

                # Convert to float64 to ensure it's the right type for time_stretch
                if isinstance(new_audio, tuple):
                    new_audio_float = new_audio[0].astype(np.float64)
                else:
                    new_audio_float = new_audio.astype(np.float64)

                # Apply time stretching
                new_audio = librosa.effects.time_stretch(
                    new_audio_float, rate=float(rate), n_fft=n_fft
                )

                # Apply a gentle envelope to ensure smooth decay
                envelope = np.ones(len(new_audio))
                fade_len = min(
                    int(middle_sr * 0.1), len(new_audio) // 10
                )  # 100ms fade or 1/10 of length

                # Only apply fade out (keep the attack intact)
                if fade_len > 0:
                    envelope[-fade_len:] = np.linspace(1, 0, fade_len)

                new_audio = new_audio * envelope

            # Normalize the final output to prevent clipping
            if np.max(np.abs(new_audio)) > 0:
                new_audio = new_audio / np.max(np.abs(new_audio)) * 0.95

            return new_audio, middle_sr

        return middle_chord_audio, middle_sr

    # For normal octaves, continue with the original approach
    # Get the sample rate from the first available sample
    first_sample = all_samples[0]

    # Check both source and target directories for the first sample
    first_path = None
    if os.path.exists(os.path.join(target_dir, first_sample)):
        first_path = os.path.join(target_dir, first_sample)
    elif os.path.exists(os.path.join(source_dir, first_sample)):
        first_path = os.path.join(source_dir, first_sample)

    # If we still can't find the file, try to find any valid sample
    if not first_path:
        for sample in all_samples:
            if os.path.exists(os.path.join(target_dir, sample)):
                first_path = os.path.join(target_dir, sample)
                break
            elif os.path.exists(os.path.join(source_dir, sample)):
                first_path = os.path.join(source_dir, sample)
                break

    if not first_path:
        print_error(
            f"Error: Could not find any valid sample files in {source_dir} or {target_dir}"
        )
        return None, None

    # Load the sample to get the sample rate
    _, sr = librosa.load(first_path, sr=None)

    # Initialize an empty array for the mixed audio
    mixed_audio = None
    max_length = 0

    # Load and process each note in the chord
    note_audios = []
    for semitone in semitones:
        # Get the note and octave for this semitone offset
        note, octave = get_note_from_semitone(root_note, root_octave, semitone)

        # Find the sample file for this note
        note_filename = f"{prefix}-{note}{octave}.wav"

        # Debug information
        print_info(f"Looking for note: {note_filename}")
        print_info(f"  Checking target dir: {os.path.join(target_dir, note_filename)}")
        print_info(f"  Checking source dir: {os.path.join(source_dir, note_filename)}")

        # Check if the file exists in target directory first, then source
        note_path = None
        if os.path.exists(os.path.join(target_dir, note_filename)):
            note_path = os.path.join(target_dir, note_filename)
            print_info(f"  Found in target dir: {note_path}")
        elif os.path.exists(os.path.join(source_dir, note_filename)):
            note_path = os.path.join(source_dir, note_filename)
            print_info(f"  Found in source dir: {note_path}")
        else:
            # If the exact note doesn't exist, find the closest available note
            print_info(f"  Note not found directly, looking for closest sample")
            closest_sample = find_closest_sample(note, octave, all_samples)
            if closest_sample:
                closest_note, closest_octave = parse_note_from_filename(closest_sample)
                print_info(
                    f"  Using closest sample: {closest_sample} ({closest_note}{closest_octave})"
                )

                # Check if the closest sample exists in target or source directory
                closest_path = None
                if os.path.exists(os.path.join(target_dir, closest_sample)):
                    closest_path = os.path.join(target_dir, closest_sample)
                    print_info(f"  Found closest sample in target dir: {closest_path}")
                elif os.path.exists(os.path.join(source_dir, closest_sample)):
                    closest_path = os.path.join(source_dir, closest_sample)
                    print_info(f"  Found closest sample in source dir: {closest_path}")
                else:
                    print_warning(f"  Closest sample file not found: {closest_sample}")
                    print_warning(
                        f"  Checked: {os.path.join(target_dir, closest_sample)}"
                    )
                    print_warning(
                        f"  Checked: {os.path.join(source_dir, closest_sample)}"
                    )
                    continue

                try:
                    # Load the closest sample
                    if closest_path:  # Ensure path is not None
                        audio, sr = librosa.load(closest_path, sr=None)
                        print_info(f"  Successfully loaded closest sample")
                    else:
                        print_error(f"  Error: closest_path is None")
                        continue

                    # Pitch shift to the target note
                    audio, sr = pitch_shift_sample(
                        audio, sr, closest_note, closest_octave, note, octave
                    )
                    print_info(f"  Successfully pitch-shifted to {note}{octave}")

                    # Trim silence at the beginning to ensure all notes start together
                    # Ensure audio is a numpy array
                    if isinstance(audio, tuple):
                        audio = audio[0]

                    # Convert to float64 for librosa.effects.trim
                    audio_float = (
                        audio.astype(np.float64) if hasattr(audio, "astype") else audio
                    )
                    audio_trimmed, _ = librosa.effects.trim(
                        audio_float, top_db=30, frame_length=512, hop_length=128
                    )

                    # Make sure we're working with a numpy array, not a tuple
                    if isinstance(audio_trimmed, tuple):
                        audio = audio_trimmed[
                            0
                        ]  # Extract the audio data from the tuple
                    else:
                        audio = audio_trimmed

                    note_audios.append(audio)
                    max_length = max(max_length, len(audio))
                    print_info(f"  Successfully processed note for chord")
                    continue
                except Exception as e:
                    print_error(f"  Error processing closest sample: {str(e)}")
                    continue
            else:
                print_warning(
                    f"  Warning: Could not find a suitable sample for {note}{octave} in chord"
                )
                continue

        try:
            # Load the audio for this note
            audio, _ = librosa.load(note_path, sr=sr)
            print_info(f"  Successfully loaded note")

            # Trim silence at the beginning to ensure all notes start together
            # Ensure audio is a numpy array
            if isinstance(audio, tuple):
                audio = audio[0]

            # Convert to float64 for librosa.effects.trim
            audio_float = (
                audio.astype(np.float64) if hasattr(audio, "astype") else audio
            )
            audio_trimmed, _ = librosa.effects.trim(
                audio_float, top_db=30, frame_length=512, hop_length=128
            )

            # Make sure we're working with a numpy array, not a tuple
            if isinstance(audio_trimmed, tuple):
                audio = audio_trimmed[0]  # Extract the audio data from the tuple
            else:
                audio = audio_trimmed

            note_audios.append(audio)
            max_length = max(max_length, len(audio))
            print_info(f"  Successfully processed note for chord")
            continue
        except Exception as e:
            print_error(f"  Error processing note: {str(e)}")
            continue

    if not note_audios:
        print(
            f"Error: Could not generate chord with root {root_note}{root_octave} - no valid notes found"
        )
        return None, None

    # Extend chord duration by the specified factor
    if chord_duration_factor > 1.0:
        # Instead of repeating the sustain, we'll use a proper time stretching approach
        # but apply it to the entire mixed chord after all notes are combined

        # First, mix the notes at their original length
        aligned_note_audios = []
        for audio in note_audios:
            # Pad shorter audio to match the longest one
            padded_audio = np.pad(audio, (0, max(0, max_length - len(audio))))
            aligned_note_audios.append(padded_audio)

        # Mix all notes together
        mixed_audio = np.zeros(max_length)
        for audio in aligned_note_audios:
            # Add to the mix (with normalization to prevent clipping)
            mixed_audio += audio / len(aligned_note_audios)

        # Now time-stretch the entire mixed chord
        # This preserves the natural envelope and harmonics better

        # For very short samples, use a smaller n_fft value
        n_fft = 2048  # Default value
        if len(mixed_audio) < n_fft:
            # Use a power of 2 that's smaller than the audio length
            n_fft = 2 ** int(np.log2(len(mixed_audio) - 1))
            n_fft = max(32, n_fft)  # Ensure it's not too small

        # Convert to float64 to ensure it's the right type for time_stretch
        mixed_audio_float = mixed_audio.astype(np.float64)

        # Use librosa's time stretching on the mixed chord
        # For stretching, rate < 1 makes it longer
        rate = 1.0 / chord_duration_factor
        mixed_audio = librosa.effects.time_stretch(
            mixed_audio_float, rate=float(rate), n_fft=n_fft
        )

        # Apply a gentle envelope to ensure smooth decay
        envelope = np.ones(len(mixed_audio))
        fade_len = min(
            int(sr * 0.1), len(mixed_audio) // 10
        )  # 100ms fade or 1/10 of length

        # Only apply fade out (keep the attack intact)
        if fade_len > 0:
            envelope[-fade_len:] = np.linspace(1, 0, fade_len)

        mixed_audio = mixed_audio * envelope

        # Normalize the final output to prevent clipping
        if np.max(np.abs(mixed_audio)) > 0:
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.95

        return mixed_audio, sr

    # If no duration extension, continue with the original approach
    # Ensure all notes have the same length
    aligned_note_audios = []
    for audio in note_audios:
        # Pad shorter audio to match the longest one
        padded_audio = np.pad(audio, (0, max(0, max_length - len(audio))))
        aligned_note_audios.append(padded_audio)

    # Mix all notes together
    mixed_audio = np.zeros(max_length)
    for audio in aligned_note_audios:
        # Add to the mix (with normalization to prevent clipping)
        mixed_audio += audio / len(aligned_note_audios)

    # Apply a slight fade-in and fade-out to prevent clicks
    fade_samples = min(int(sr * 0.01), max_length // 10)  # 10ms fade or 1/10 of length
    if fade_samples > 0:
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        mixed_audio[:fade_samples] *= fade_in

        # Fade out
        fade_out = np.linspace(1, 0, fade_samples)
        mixed_audio[-fade_samples:] *= fade_out

    # Normalize the final output to prevent clipping
    if np.max(np.abs(mixed_audio)) > 0:
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.95

    return mixed_audio, sr


def generate_chords(
    prefix,
    all_samples,
    source_dir,
    chord_dir,
    target_dir,
    chord_qualities=None,
    generate_inversions=False,
    selected_chord_types=None,  # New parameter for selected chord types
    selected_inversions=None,  # New parameter for selected inversions
    overwrite=False,
):
    """Generate chord samples based on the provided chord definitions."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Check if chord_dir already exists and overwrite is not enabled
    if not overwrite and os.path.exists(chord_dir):
        print_info(
            f"Chord directory {chord_dir} already exists and overwrite is not enabled. Skipping chord generation."
        )
        return chord_dir, []

    # Use the hardcoded chord definitions
    chord_defs = CHORD_DEFINITIONS

    # Acquire lock for consistent console output
    with tqdm_lock:
        # Filter chord definitions by quality if specified
        if chord_qualities:
            chord_defs = [chord for chord in chord_defs if chord[1] in chord_qualities]
            print_info(
                f"Filtered to {len(chord_defs)} chord types in qualities: {', '.join(chord_qualities)}"
            )

        # Further filter by selected chord types if specified
        if selected_chord_types:
            filtered_defs = []
            for chord_name, quality, semitones, notes_count in chord_defs:
                if (
                    quality in selected_chord_types
                    and chord_name in selected_chord_types[quality]
                ):
                    filtered_defs.append((chord_name, quality, semitones, notes_count))

            chord_defs = filtered_defs
            print_info(f"Further filtered to {len(chord_defs)} specific chord types")

        print_header(f"Generating {len(chord_defs)} chord types")
        if generate_inversions:
            if selected_inversions is not None:
                print_info(
                    f"Generating {len(selected_inversions)} selected inversions: {', '.join(str(i) for i in selected_inversions)}"
                )
            else:
                print_info("Inversions will be generated for all applicable chords")

        # Create the main chord directory if it doesn't exist
        if not os.path.exists(chord_dir):
            os.makedirs(chord_dir)

        # Group chord definitions by quality
        chord_by_quality = {}
        for chord_name, quality, semitones, _ in chord_defs:
            if quality not in chord_by_quality:
                chord_by_quality[quality] = []
            chord_by_quality[quality].append((chord_name, semitones))

        # Count total chords to generate for progress tracking
        total_chords = 0
        for quality, chords in chord_by_quality.items():
            for _, semitones in chords:
                # Count chords for all octaves (1-8)
                for octave in range(1, 9):
                    for note in notes:
                        # Skip if the highest note in the chord would be above B8
                        highest_semitone = max(semitones)
                        highest_note, highest_octave = get_note_from_semitone(
                            note, octave, highest_semitone
                        )
                        if highest_octave <= 8:
                            total_chords += 1

                            # Add count for inversions if enabled
                            if generate_inversions and len(semitones) >= 3:
                                # Add inversions count (number of inversions = number of notes - 1)
                                total_chords += len(semitones) - 1

        # Set up progress bars with fixed positions but offset for thread safety
        # Use higher positions to avoid conflicts with other functions' progress bars
        base_position = 3  # Start at position 3 to avoid conflicts

        # Create a master progress bar at the bottom position
        master_pbar = tqdm(
            total=total_chords,
            desc=f"Overall chord progress for {os.path.basename(source_dir)}",
            position=base_position + 2,  # Fixed position at the bottom
            leave=True,
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, RESET),
        )

        # Create a quality progress bar above the master bar
        quality_pbar = tqdm(
            total=len(chord_by_quality),
            desc="Chord quality progress",
            position=base_position + 1,  # Fixed position above master bar
            leave=True,
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, RESET),
        )

    # Generate chords for each quality
    for quality_idx, (quality, chords) in enumerate(chord_by_quality.items()):
        with tqdm_lock:
            # Create a directory for this chord quality
            quality_dir = os.path.join(chord_dir, quality)
            if not os.path.exists(quality_dir):
                os.makedirs(quality_dir)

            # Create inversions directory if needed
            if generate_inversions:
                inversions_dir = os.path.join(quality_dir, "inv")
                if not os.path.exists(inversions_dir):
                    os.makedirs(inversions_dir)

            # Update the current task message
            tqdm.write(
                f"{INFO}Generating {len(chords)} {quality} chord types...{RESET}"
            )

            # Create a chord type progress bar
            chord_pbar = tqdm(
                total=len(chords),
                desc=f"{quality} chords",
                position=base_position,  # Top position for current task
                leave=False,  # Don't leave this bar when done
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, RESET),
            )

        # Generate each chord type with roots from C1 to B8
        for chord_idx, (chord_name, semitones) in enumerate(chords):
            # Create a safe filename from the chord name
            safe_chord_name = chord_to_filename(chord_name)

            # Update chord progress
            with tqdm_lock:
                chord_pbar.set_description(f"Generating {chord_name}")

            # First generate the core chords (C2-B4) directly
            core_chords = {}

            # Generate inversions if requested and applicable
            inversions = []
            if generate_inversions and len(semitones) >= 3:
                inversions = generate_chord_inversions(semitones)
                if inversions:
                    tqdm.write(
                        f"{INFO}    Will generate {len(inversions)} inversions for {chord_name}{RESET}"
                    )

            # Generate core chords with roots from C2 to B4
            for octave in range(2, 5):
                for note in notes:
                    # Skip if the highest note in the chord would be above B8
                    highest_semitone = max(semitones)
                    highest_note, highest_octave = get_note_from_semitone(
                        note, octave, highest_semitone
                    )
                    if highest_octave > 8:
                        continue

                    # Generate the chord
                    chord_audio, sr = generate_chord(
                        note,
                        octave,
                        semitones,
                        all_samples,
                        source_dir,
                        target_dir,
                        prefix,
                        chord_duration_factor=4.0,
                    )

                    if chord_audio is not None:
                        # Save the chord
                        chord_filename = f"{safe_chord_name}-{note}{octave}.wav"
                        chord_path = os.path.join(quality_dir, chord_filename)
                        sf.write(chord_path, chord_audio, sr)
                        tqdm.write(f"{SUCCESS}    Generated {chord_filename}{RESET}")

                        # Store the chord for later use in pitch shifting
                        core_chords[(note, octave)] = (chord_path, chord_audio, sr)

                        # Generate inversions for this chord if requested
                        if generate_inversions and inversions:
                            for inv_num, inv_semitones in inversions:
                                # Skip if this inversion is not in the selected inversions for this chord
                                if selected_inversions is not None:
                                    # Check if this chord type has selected inversions
                                    chord_key = (quality, chord_name)
                                    if (
                                        chord_key not in selected_inversions
                                        or inv_num not in selected_inversions[chord_key]
                                    ):
                                        continue

                                try:
                                    # Compute the new root for the inversion based on the original semitones.
                                    # The inversion index inv_num indicates which note becomes the new root.
                                    new_root, new_octave = get_note_from_semitone(
                                        note, octave, semitones[inv_num]
                                    )

                                    # Add additional debug log
                                    tqdm.write(
                                        f"{INFO}    Generating inversion {inv_num} for {chord_name} with new root {new_root}{new_octave}{RESET}"
                                    )

                                    # Try to generate the inverted chord using the inverted semitones and new root
                                    inv_chord_audio, inv_sr = generate_chord(
                                        new_root,
                                        new_octave,
                                        inv_semitones,
                                        all_samples,
                                        source_dir,
                                        target_dir,
                                        prefix,
                                        chord_duration_factor=4.0,
                                    )

                                    # Check if inversion generation returned None
                                    if inv_chord_audio is None:
                                        tqdm.write(
                                            f"{WARNING}    Inversion generation returned None for inversion {inv_num} of {chord_name} at {new_root}{new_octave}{RESET}"
                                        )
                                        raise ValueError("Inversion generation failed")

                                    # Save the inverted chord with inversion number in filename
                                    inv_chord_filename = f"{safe_chord_name}-{inv_num}stInv-{new_root}{new_octave}.wav"
                                    inv_chord_path = os.path.join(
                                        inversions_dir, inv_chord_filename
                                    )
                                    sf.write(inv_chord_path, inv_chord_audio, inv_sr)
                                    tqdm.write(
                                        f"{SUCCESS}    Generated {inv_chord_filename}{RESET}"
                                    )
                                except Exception as e:
                                    # If there's an error, fall back to using the original chord audio
                                    tqdm.write(
                                        f"{WARNING}    Error generating inversion {inv_num} for {chord_name}: {str(e)}{RESET}"
                                    )

                                    # Save the inverted chord with inversion number in filename using the original chord audio as fallback
                                    inv_chord_filename = f"{safe_chord_name}-{inv_num}stInv-{note}{octave}.wav"
                                    inv_chord_path = os.path.join(
                                        inversions_dir, inv_chord_filename
                                    )
                                    sf.write(inv_chord_path, chord_audio, sr)
                                    tqdm.write(
                                        f"{SUCCESS}    Generated {inv_chord_filename} (fallback){RESET}"
                                    )

            # Now generate the extended range (C1-B1 and C5-B8) by pitch shifting
            # First, find the closest core chord for each target chord
            for octave in range(1, 9):
                # Skip the core octaves we've already generated
                if 2 <= octave <= 4:
                    continue

                for note in notes:
                    # Skip if the highest note in the chord would be above B8
                    highest_semitone = max(semitones)
                    highest_note, highest_octave = get_note_from_semitone(
                        note, octave, highest_semitone
                    )
                    if highest_octave > 8:
                        continue

                    # Find the closest core chord to use as source
                    closest_core = None
                    min_distance = float("inf")

                    for core_note, core_octave in core_chords.keys():
                        # Calculate semitone distance
                        core_index = notes.index(core_note) + (core_octave * 12)
                        target_index = notes.index(note) + (octave * 12)
                        distance = abs(target_index - core_index)

                        if distance < min_distance:
                            min_distance = distance
                            closest_core = (core_note, core_octave)

                    if closest_core:
                        source_note, source_octave = closest_core
                        chord_path, chord_audio, sr = core_chords[closest_core]

                        # Pitch shift the chord
                        with tqdm_lock:
                            tqdm.write(
                                f"{INFO}    Pitch shifting {source_note}{source_octave} chord to {note}{octave}...{RESET}"
                            )

                        # Calculate pitch shift factor based on semitone distance
                        source_freq = get_note_frequency(source_note, source_octave)
                        target_freq = get_note_frequency(note, octave)
                        shift_factor = target_freq / source_freq

                        # Resample the audio
                        if shift_factor != 1.0:
                            # For higher notes, we need to shorten the sample (speed up)
                            # For lower notes, we need to lengthen the sample (slow down)
                            new_length = int(len(chord_audio) / shift_factor)
                            new_audio = scipy_signal.resample(chord_audio, new_length)

                            # Time stretching to maintain consistent duration
                            # First, calculate the current duration
                            current_duration = len(new_audio) / sr

                            # Get the duration of the source chord
                            source_duration = len(chord_audio) / sr

                            # Calculate stretch factor to match the source duration
                            stretch_factor = source_duration / current_duration

                            # Only apply time stretching if the difference is significant
                            if abs(stretch_factor - 1.0) > 0.01:
                                with tqdm_lock:
                                    tqdm.write(
                                        f"{INFO}    Time stretching to match duration (factor: {stretch_factor:.2f}){RESET}"
                                    )

                                # For librosa's time_stretch, rate < 1 makes it longer
                                rate = 1.0 / stretch_factor

                                # For very short samples, use a smaller n_fft value
                                n_fft = 2048  # Default value
                                if len(new_audio) < n_fft:
                                    # Use a power of 2 that's smaller than the audio length
                                    n_fft = 2 ** int(np.log2(len(new_audio) - 1))
                                    n_fft = max(32, n_fft)  # Ensure it's not too small

                                # Convert to float64 to ensure it's the right type for time_stretch
                                if isinstance(new_audio, tuple):
                                    new_audio_float = new_audio[0].astype(np.float64)
                                else:
                                    new_audio_float = new_audio.astype(np.float64)

                                # Apply time stretching
                                new_audio = librosa.effects.time_stretch(
                                    new_audio_float, rate=float(rate), n_fft=n_fft
                                )

                                # Apply a gentle envelope to ensure smooth decay
                                envelope = np.ones(len(new_audio))
                                fade_len = min(
                                    int(sr * 0.1), len(new_audio) // 10
                                )  # 100ms fade or 1/10 of length

                                # Only apply fade out (keep the attack intact)
                                if fade_len > 0:
                                    envelope[-fade_len:] = np.linspace(1, 0, fade_len)

                                new_audio = new_audio * envelope

                            # Normalize the final output to prevent clipping
                            if np.max(np.abs(new_audio)) > 0:
                                new_audio = new_audio / np.max(np.abs(new_audio)) * 0.95

                            # Save the pitch-shifted chord
                            chord_filename = f"{safe_chord_name}-{note}{octave}.wav"
                            chord_path = os.path.join(quality_dir, chord_filename)
                            sf.write(chord_path, new_audio, sr)

                            with tqdm_lock:
                                tqdm.write(
                                    f"{SUCCESS}    Generated {chord_filename} (pitch-shifted){RESET}"
                                )

                            # Generate inversions for this chord if requested
                            if generate_inversions and inversions:
                                for inv_num, inv_semitones in inversions:
                                    # For pitch-shifted chords, we need to use the same approach
                                    # Instead of generating from scratch, pitch-shift the chord we just created
                                    inv_chord_filename = f"{safe_chord_name}-{inv_num}stInv-{note}{octave}.wav"
                                    inv_chord_path = os.path.join(
                                        inversions_dir, inv_chord_filename
                                    )

                                    # We already have the new_audio, so we'll use that
                                    # This avoids trying to access files that might not exist
                                    sf.write(inv_chord_path, new_audio, sr)

                                    with tqdm_lock:
                                        tqdm.write(
                                            f"{SUCCESS}    Generated {inv_chord_filename} (pitch-shifted){RESET}"
                                        )

                    # Update the master progress bar
                    with tqdm_lock:
                        master_pbar.update(1)

                        # Update for inversions if applicable
                        if generate_inversions and inversions and closest_core:
                            master_pbar.update(len(inversions))

            # Update chord progress bar
            with tqdm_lock:
                chord_pbar.update(1)

        # Close the chord progress bar
        with tqdm_lock:
            chord_pbar.close()

        # Update quality progress bar
        with tqdm_lock:
            quality_pbar.update(1)

    # Close progress bars
    with tqdm_lock:
        quality_pbar.close()
        master_pbar.close()

        tqdm.write(f"{SUCCESS}\nChord generation complete!{RESET}")

    # Generate full chord sample files by type
    full_chord_filenames = generate_full_chord_samples(chord_dir, prefix)

    # Do final updates before returning
    with tqdm_lock:
        # Close the quality progress bar
        quality_pbar.update(1)

    # Generate full chord samples
    print_info("Generating full chord samples...")
    full_chord_filenames = generate_full_chord_samples(chord_dir, prefix)
    print_info(f"Generated {len(full_chord_filenames)} full chord sample files")

    # Debug info before returning
    print_info(f"Returning chord_dir: {chord_dir}")
    print_info(f"Number of full_chord_filenames: {len(full_chord_filenames)}")
    for idx, item in enumerate(full_chord_filenames):
        print_info(f"  Chord file {idx+1}: {item}")

    return chord_dir, full_chord_filenames


def generate_missing_samples(
    prefix, existing_samples, source_dir, target_dir, time_match=False
):
    """Generate all missing samples across the 8-octave range."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    generated_files = []

    # Parse existing samples
    parsed_samples = {}
    for sample in existing_samples:
        note, octave = parse_note_from_filename(sample)
        if note and octave:
            parsed_samples[sample] = (note, octave)

    # Acquire lock for consistent output
    with tqdm_lock:
        # Calculate average duration of existing samples more accurately
        durations = []
        for sample in existing_samples:
            if sample in parsed_samples:
                # Get file info directly using soundfile instead of loading the entire file
                file_path = os.path.join(source_dir, sample)
                info = sf.info(file_path)
                duration = info.duration  # This gives the exact duration in seconds
                durations.append(duration)
                print_info(f"Sample {sample} duration: {duration:.2f} seconds")

        avg_duration = np.mean(durations) if durations else 2.0  # Default to 2 seconds

        if time_match and durations:
            print_info(
                f"Time matching enabled - target duration: {avg_duration:.2f} seconds"
            )

        # Count total samples to generate
        total_to_generate = 0
        for octave in range(1, 9):
            for note in notes:
                target_filename = f"{prefix}-{note}{octave}.wav"
                if target_filename not in existing_samples:
                    total_to_generate += 1

        # Create progress bar at a fixed position - use position 1 to avoid conflicts with other progress bars
        position = 1
        pbar = tqdm(
            total=total_to_generate,
            desc=f"Generating samples for {os.path.basename(source_dir)}",
            position=position,
            leave=True,
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, RESET),
        )

    # Update status
    update_status(
        source_dir, f"Preparing to generate {total_to_generate} missing samples", "info"
    )

    # Track progress for status updates
    completed_samples = 0
    last_status_update_percent = 0

    # Generate missing samples for all notes in octaves 1-8
    for octave in range(1, 9):
        for note in notes:
            target_filename = f"{prefix}-{note}{octave}.wav"

            # Skip if file already exists in source directory
            if target_filename in existing_samples:
                generated_files.append(target_filename)
                continue

            # Skip if file already exists in target directory (if different from source)
            if source_dir != target_dir and os.path.exists(
                os.path.join(target_dir, target_filename)
            ):
                generated_files.append(target_filename)
                continue

            # Find closest sample to use as source
            source_sample = find_closest_sample(note, octave, existing_samples)
            if not source_sample:
                with tqdm_lock:
                    print_warning(
                        f"No suitable source sample found for {target_filename}"
                    )
                continue

            source_note, source_octave = parsed_samples[source_sample]

            # Load source audio
            audio, sr = librosa.load(os.path.join(source_dir, source_sample), sr=None)

            # Pitch shift to target note
            new_audio, new_sr = pitch_shift_sample(
                audio, sr, source_note, source_octave, note, octave
            )

            # Time stretch to match average duration if requested
            if time_match and durations:
                # Calculate current duration directly from the audio data
                current_duration = len(new_audio) / new_sr
                with tqdm_lock:
                    print_info(
                        f"  Before stretching: {target_filename} duration: {current_duration:.2f} seconds"
                    )

                # In librosa's time_stretch, rate > 1 speeds up, rate < 1 slows down
                # So we need to use 1/stretch_factor to get the correct behavior
                stretch_factor = avg_duration / current_duration
                rate = 1.0 / stretch_factor  # Inverse for librosa's time_stretch

                if (
                    abs(stretch_factor - 1.0) > 0.01
                ):  # Only stretch if difference is significant
                    with tqdm_lock:
                        print_info(
                            f"  Time stretching {target_filename} (target: {avg_duration:.2f}s, factor: {stretch_factor:.2f}, rate: {rate:.2f})"
                        )

                    # For very short samples, use a smaller n_fft value
                    n_fft = 2048  # Default value
                    if len(new_audio) < n_fft:
                        # Use a power of 2 that's smaller than the audio length
                        n_fft = 2 ** int(np.log2(len(new_audio) - 1))
                        n_fft = max(32, n_fft)  # Ensure it's not too small
                        with tqdm_lock:
                            print_info(
                                f"  Using smaller FFT window (n_fft={n_fft}) for short sample"
                            )

                    # Use librosa's high-quality time stretching with corrected rate and appropriate n_fft
                    # Convert to float64 to ensure correct type for time_stretch
                    # Make sure new_audio is a numpy array, not a tuple
                    if isinstance(new_audio, tuple):
                        new_audio_float = new_audio[0].astype(np.float64)
                    else:
                        new_audio_float = new_audio.astype(np.float64)

                    new_audio = librosa.effects.time_stretch(
                        new_audio_float, rate=float(rate), n_fft=n_fft
                    )

                    # Apply a gentle envelope to ensure smooth decay
                    envelope = np.ones(len(new_audio))
                    fade_len = min(
                        int(new_sr * 0.1), len(new_audio) // 10
                    )  # 100ms fade or 1/10 of length

                    # Only apply fade out (keep the attack intact)
                    if fade_len > 0:
                        envelope[-fade_len:] = np.linspace(1, 0, fade_len)

                    new_audio = new_audio * envelope

            # Normalize audio to prevent clipping
            if np.max(np.abs(new_audio)) > 0:
                new_audio = new_audio / np.max(np.abs(new_audio)) * 0.95

            # Save the pitch-shifted and time-stretched audio
            output_path = os.path.join(target_dir, target_filename)
            sf.write(output_path, new_audio, new_sr)

            # Add the generated file to the list
            generated_files.append(target_filename)

            # Print success message
            with tqdm_lock:
                print_success(f"Generated {target_filename}")

            # Update the progress bar
            with tqdm_lock:
                pbar.update(1)

            # Update progress tracking
            completed_samples += 1
            progress_percent = (
                int((completed_samples / total_to_generate) * 100)
                if total_to_generate > 0
                else 100
            )

            # Update status message at 10% increments
            if (
                progress_percent >= last_status_update_percent + 10
                or completed_samples == total_to_generate
            ):
                last_status_update_percent = progress_percent
                update_status(
                    source_dir,
                    f"Generated {completed_samples}/{total_to_generate} samples ({progress_percent}%)",
                    "info",
                )

    # Close the progress bar
    with tqdm_lock:
        pbar.close()

    return generated_files


def play_all_notes(all_samples, source_dir, target_dir):
    """Play all notes in sequence from lowest to highest."""
    # Sort samples by note and octave
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def sort_key(filename):
        note, octave = parse_note_from_filename(filename)
        if note and octave:
            return octave * 100 + notes.index(note)
        return 0

    sorted_samples = sorted(all_samples, key=sort_key)

    print("\nPlaying all notes in sequence...")
    for sample in sorted_samples:
        note, octave = parse_note_from_filename(sample)
        if note and octave:
            print(f"Playing {sample} ({note}{octave})")

            # First check if the file exists in the target directory
            if os.path.exists(os.path.join(target_dir, sample)):
                file_path = os.path.join(target_dir, sample)
            else:
                # Otherwise use the source directory
                file_path = os.path.join(source_dir, sample)

            audio, sr = librosa.load(file_path, sr=None)
            sd.play(audio, sr)
            sd.wait()  # Wait until playback is finished


def generate_full_sample(all_samples, prefix, source_dir, target_dir):
    """Generate a single WAV file with all notes in sequence and embedded slice markers."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    import struct
    import wave

    def sort_key(filename):
        note, octave = parse_note_from_filename(filename)
        if note and octave:
            return octave * 100 + notes.index(note)
        return 0

    sorted_samples = sorted(all_samples, key=sort_key)

    # Get sample rate from first file (assuming all files have the same sample rate)
    first_sample = sorted_samples[0]
    first_path = os.path.join(source_dir, first_sample)
    if not os.path.exists(first_path):
        first_path = os.path.join(target_dir, first_sample)
    _, sr = librosa.load(first_path, sr=None)

    # Create silence between notes (0.5 seconds)
    silence_duration = 0.5  # seconds
    silence_samples = int(silence_duration * sr)
    silence = np.zeros(silence_samples)

    # Combine all samples with silence in between
    combined_audio = np.array([])

    # Track positions for slice markers
    cue_positions = []  # in samples
    current_position = 0  # in samples

    print("\nGenerating full sample file...")
    for sample in sorted_samples:
        note, octave = parse_note_from_filename(sample)
        if note and octave:
            print(f"Adding {sample} ({note}{octave}) to full sample")

            # Check if file exists in target directory first, then source
            if os.path.exists(os.path.join(target_dir, sample)):
                file_path = os.path.join(target_dir, sample)
            else:
                file_path = os.path.join(source_dir, sample)

            audio, _ = librosa.load(file_path, sr=sr)

            # Detect and trim silence at the beginning and end
            # This helps prevent clicks without altering the actual sound
            # Ensure audio is a numpy array
            if isinstance(audio, tuple):
                audio = audio[0]

            # Convert to float64 for librosa.effects.trim
            audio_float = (
                audio.astype(np.float64) if hasattr(audio, "astype") else audio
            )
            non_silent = librosa.effects.trim(
                audio_float,
                top_db=40,  # Higher value = more aggressive trimming
                frame_length=512,
                hop_length=128,
            )[0]

            # Record the start position of this note (in samples)
            cue_positions.append((f"{note}{octave}", current_position))

            # Add the audio and silence
            combined_audio = np.append(combined_audio, non_silent)
            current_position += len(non_silent)  # Update position after audio

            combined_audio = np.append(combined_audio, silence)
            current_position += silence_samples  # Update position after silence

    # Save the combined audio to the exp directory in the source directory
    exp_dir = os.path.join(source_dir, "exp")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    output_filename = f"{prefix}.wav"
    output_path = os.path.join(exp_dir, output_filename)

    # First save the audio data
    sf.write(output_path, combined_audio, sr)
    print(f"Generated full sample file: {output_filename} (in exp directory)")

    # Now add slice markers to the WAV file
    try:
        # Open the WAV file for reading and writing in binary mode
        with wave.open(output_path, "rb") as wav_read:
            params = wav_read.getparams()
            frames = wav_read.readframes(wav_read.getnframes())

        # Create a new WAV file with slice markers
        with wave.open(output_path + ".temp", "wb") as wav_write:
            wav_write.setparams(params)

            # Write the audio data
            wav_write.writeframes(frames)

            # Add cue chunk
            cue_chunk = create_cue_chunk(cue_positions)

            # We need to manually add the cue chunk to the file
            # This is a bit hacky but necessary since wave module doesn't support cue chunks
            with open(output_path + ".temp", "ab") as f:
                f.write(b"cue ")  # Chunk ID
                f.write(struct.pack("<I", len(cue_chunk)))  # Chunk size
                f.write(cue_chunk)  # Chunk data

        # Replace the original file with the new one
        os.replace(output_path + ".temp", output_path)
        print(f"Added {len(cue_positions)} slice markers to {output_filename}")

    except Exception as e:
        print(f"Warning: Could not add slice markers to WAV file: {e}")

    return output_filename


def create_cue_chunk(cue_positions):
    """Create a cue chunk for a WAV file."""
    import struct

    # Number of cue points
    num_cues = len(cue_positions)

    # Start with the number of cue points
    chunk_data = struct.pack("<I", num_cues)

    # Add each cue point
    for i, (label, position) in enumerate(cue_positions):
        cue_id = i + 1
        position_samples = position

        # Cue point structure:
        # ID (4 bytes) + Position (4 bytes) + Data Chunk ID (4 bytes) +
        # Chunk Start (4 bytes) + Block Start (4 bytes) + Sample Offset (4 bytes)
        cue_point = struct.pack(
            "<II4sIII",
            cue_id,  # ID
            position_samples,  # Position
            b"data",  # Data Chunk ID
            0,  # Chunk Start
            0,  # Block Start
            position_samples,
        )  # Sample Offset

        chunk_data += cue_point

    return chunk_data


def cleanup_artifacts(output_dir, chord_dirs=None, full_chord_filenames=None):
    """Cleanup temporary files and organize the final output."""
    print_header("Cleaning up and organizing final output")
    
    # Create exp/chords directory to copy chord files to
    chords_dir = os.path.join(output_dir, "chords")
    if not os.path.exists(chords_dir):
        os.makedirs(chords_dir)
    
    # Copy chord files to their respective quality directories in exp/chords
    if full_chord_filenames and chord_dirs:
        print_info(f"Found {len(full_chord_filenames)} chord files to copy to {chords_dir}")
        success_count = 0
        
        # Helper function to find a file by trying different path formats
        def find_chord_file(chord_item, chord_dirs):
            paths_to_try = []
            
            if isinstance(chord_item, tuple):
                # Handle tuple format (quality, subdir?, filename)
                if len(chord_item) == 3:
                    quality, subdir, filename = chord_item
                    for chord_dir in chord_dirs:
                        paths_to_try.append(os.path.join(chord_dir, quality, subdir, filename))
                else:
                    quality, filename = chord_item
                    for chord_dir in chord_dirs:
                        paths_to_try.append(os.path.join(chord_dir, quality, filename))
            else:
                # Handle string format (filename only)
                filename = chord_item
                for chord_dir in chord_dirs:
                    paths_to_try.append(os.path.join(chord_dir, filename))
            
            # Try all possible paths
            for path in paths_to_try:
                if os.path.exists(path):
                    return path, filename
            
            # Try secondary fallback paths with different formats
            if isinstance(chord_item, tuple) and len(chord_item) >= 2:
                quality = chord_item[0]
                filename = chord_item[-1]
                for chord_dir in chord_dirs:
                    # Try without subdirectory
                    paths_to_try.append(os.path.join(chord_dir, quality, filename))
                    # Try with chord directory
                    paths_to_try.append(os.path.join(chord_dir, "chord", quality, filename))
                
                for path in paths_to_try:
                    if os.path.exists(path):
                        return path, filename
            
            return None, None
            
        # Process each chord file
        for chord_item in full_chord_filenames:
            chord_path, filename = find_chord_file(chord_item, chord_dirs)
            
            if chord_path:
                print_info(f"Found chord file at: {chord_path}")
                
                # Determine if this is an inversion
                is_inversion = (
                    "inv" in chord_path
                    or "inversions" in chord_path
                    or ("filename" in locals() and "-stInv-" in filename)
                    or ("filename" in locals() and re.search(r"-\d+stInv-", filename) is not None)
                )
                
                # Extract quality from chord_item or from chord_path
                if isinstance(chord_item, tuple) and len(chord_item) >= 1:
                    quality = chord_item[0]
                else:
                    # Extract quality from the filename or chord_path
                    quality_match = re.search(r"/([^/]+)/(?:inv|inversions)?/[^/]+\.wav$", chord_path)
                    if quality_match:
                        quality = quality_match.group(1)
                    else:
                        quality_match = re.search(r"-([^-]+)(?:(?:-\d+)?-Full)?\.wav$", 
                                                os.path.basename(chord_path))
                        if quality_match:
                            quality = quality_match.group(1)
                        else:
                            quality = "Other"
                
                # Create quality directory in exp/chords
                quality_dir = os.path.join(chords_dir, quality)
                if not os.path.exists(quality_dir):
                    os.makedirs(quality_dir)
                    print_info(f"Created quality directory: {quality_dir}")
                
                # Create inversions directory if needed
                dest_dir = quality_dir
                if is_inversion:
                    inversions_dir = os.path.join(quality_dir, "inv")
                    if not os.path.exists(inversions_dir):
                        os.makedirs(inversions_dir)
                        print_info(f"Created inversions directory: {inversions_dir}")
                    dest_dir = inversions_dir
                
                # Get the destination filename (clean up any "-Full" suffix)
                dest_filename = os.path.basename(chord_path)
                dest_filename = re.sub(r"(?:-\d+)?-Full(?=\.wav$)", "", dest_filename)
                dest_path = os.path.join(dest_dir, dest_filename)
                
                # Copy the file
                import shutil
                try:
                    shutil.copy2(chord_path, dest_path)

                    shutil.rmtree(target_dir)

                # Add task parameters to list
                processing_tasks.append(
                    {
                        "source_dir": directory,
                        "target_dir": target_dir,
                        "prefix": prefix,
                        "play": False,  # Disable play in parallel mode for safety
                        "gen_full": options_dict["gen_full"],
                        "time_match": options_dict["time_match"],
                        "chords": options_dict["chords"],
                        "keep_artifacts": options_dict["keep_artifacts"],
                        "chord_qualities": chord_qualities,
                        "generate_inversions": generate_inversions,
                        "selected_chord_types": selected_chord_types,  # Add new parameter
                        "selected_inversions": selected_inversions,  # Add new parameter
                        "overwrite": options_dict["overwrite"],
                    }
                )

            # Start a heartbeat thread to provide periodic updates on processing status
            heartbeat_stop = threading.Event()
            heartbeat_thread = threading.Thread(
                target=status_heartbeat,
                args=(heartbeat_stop, directories, 10),  # Update every 10 seconds
            )
            heartbeat_thread.daemon = True
            heartbeat_thread.start()

            # Process directories in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit all tasks
                futures = [
                    executor.submit(process_directory_wrapper, **task)
                    for task in processing_tasks
                ]

                # Create a progress bar to track overall completion
                with tqdm(
                    total=len(futures),
                    desc="Processing directories",
                    position=0,
                    leave=True,
                ) as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            # Get the result (will raise exception if the task failed)
                            result = future.result()
                            pbar.update(1)
                        except Exception as e:
                            print_error(f"Error in worker thread: {e}")
                            pbar.update(1)

            # Stop the heartbeat thread
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)

            # Print final summary
            print_header("Processing Summary")
            for directory in directories:
                status = processing_status.get(directory, {})
                status_msg = status.get("message", "Unknown status")
                status_type = status.get("type", "info")
                status_time = status.get("timestamp", "Unknown time")

                if status_type == "success":
                    print_success(
                        f"{os.path.basename(directory)}: {status_msg} at {status_time}"
                    )
                elif status_type == "error":
                    print_error(
                        f"{os.path.basename(directory)}: {status_msg} at {status_time}"
                    )
                elif status_type == "warning":
                    print_warning(
                        f"{os.path.basename(directory)}: {status_msg} at {status_time}"
                    )
                else:
                    print_info(
                        f"{os.path.basename(directory)}: {status_msg} at {status_time}"
                    )

            # Play all notes if requested after all processing is complete
            if options_dict["play"]:
                print_info(
                    "Processing complete. Playing samples from the first directory..."
                )
                first_dir = directories[0]
                first_target_dir = os.path.join(first_dir, "expansion")

                # Get all samples in the expansion directory
                all_samples = get_all_wav_files(first_target_dir)
                play_all_notes(all_samples, first_dir, first_target_dir)
        else:
            # Process each directory sequentially (original behavior)
            for directory in directories:
                # Create expansion subdirectory for output
                target_dir = os.path.join(directory, "expansion")

                # Delete existing expansion directory if overwrite is enabled
                if options_dict["overwrite"] and os.path.exists(target_dir):
                    print_info(f"Removing existing expansion directory: {target_dir}")
                    import shutil

                    shutil.rmtree(target_dir)

                process_directory(
                    source_dir=directory,
                    target_dir=target_dir,
                    prefix=prefix,
                    play=options_dict["play"],
                    gen_full=options_dict["gen_full"],
                    time_match=options_dict["time_match"],
                    chords=options_dict["chords"],
                    keep_artifacts=options_dict["keep_artifacts"],
                    chord_qualities=chord_qualities,
                    generate_inversions=generate_inversions,
                    selected_chord_types=selected_chord_types,  # Add new parameter
                    selected_inversions=selected_inversions,  # Add new parameter
                    overwrite=options_dict["overwrite"],
                )
    else:
        # Process just the single directory (no parallelization needed)
        # Create expansion subdirectory for output
        target_dir = os.path.join(source_dir, "expansion")

        # Delete existing expansion directory if overwrite is enabled
        if options_dict["overwrite"] and os.path.exists(target_dir):
            print_info(f"Removing existing expansion directory: {target_dir}")
            import shutil

            shutil.rmtree(target_dir)

        process_directory(
            source_dir=source_dir,
            target_dir=target_dir,
            prefix=prefix,
            play=options_dict["play"],
            gen_full=options_dict["gen_full"],
            time_match=options_dict["time_match"],
            chords=options_dict["chords"],
            keep_artifacts=options_dict["keep_artifacts"],
            chord_qualities=chord_qualities,
            generate_inversions=generate_inversions,
            selected_chord_types=selected_chord_types,  # Add new parameter
            selected_inversions=selected_inversions,  # Add new parameter
            overwrite=options_dict["overwrite"],
        )

    print_success("Processing complete!")


def process_directory_wrapper(**kwargs):
    """Wrapper for process_directory to handle thread safety and exceptions."""
    source_dir = kwargs.get("source_dir", "unknown")
    try:
        # Log start of processing
        update_status(source_dir, "Started processing", "info")

        # Process the directory
        result = process_directory(
            source_dir=kwargs.get("source_dir", "."),
            target_dir=kwargs.get("target_dir", "./expansion"),
            prefix=kwargs.get("prefix"),
            play=kwargs.get("play", False),
            gen_full=kwargs.get("gen_full", False),
            time_match=kwargs.get("time_match", False),
            chords=kwargs.get("chords", False),
            keep_artifacts=kwargs.get("keep_artifacts", False),
            chord_qualities=kwargs.get("chord_qualities"),
            generate_inversions=kwargs.get("generate_inversions", False),
            selected_chord_types=kwargs.get(
                "selected_chord_types"
            ),  # Add new parameter
            selected_inversions=kwargs.get("selected_inversions"),  # Add new parameter
            overwrite=kwargs.get("overwrite", False),
        )

        # Log successful completion
        update_status(source_dir, "Processing completed successfully", "success")

        return result
    except Exception as e:
        # Log error
        error_message = f"Error processing directory: {str(e)}"
        update_status(source_dir, error_message, "error")

        # Still raise the exception for the executor to handle
        raise e


def main():
    """Main entry point for the script."""
    # Always run in interactive mode
    interactive_mode()
    return


def status_heartbeat(stop_event, directories, interval=10):
    """Periodically print status updates for all directories being processed.

    Args:
        stop_event: A threading.Event that signals when to stop the heartbeat
        directories: List of directories being processed
        interval: How often to print updates (in seconds)
    """
    while not stop_event.is_set():
        # Wait for the specified interval, but check stop_event more frequently
        for _ in range(interval):
            if stop_event.is_set():
                return
            time.sleep(1)

        # Print a status update for all directories
        with status_lock:
            print("\n" + "=" * 80)
            print(f"{INFO}STATUS UPDATE AT {time.strftime('%H:%M:%S')}{RESET}")
            print("=" * 80)

            active_count = 0
            completed_count = 0
            error_count = 0

            for directory in directories:
                dir_name = os.path.basename(directory)
                status = processing_status.get(directory, {})

                if not status:
                    print(f"{WARNING}{dir_name}: Waiting to start{RESET}")
                    continue

                message = status.get("message", "Unknown")
                timestamp = status.get("timestamp", "Unknown")
                status_type = status.get("type", "info")

                if status_type == "success":
                    print(f"{SUCCESS}{dir_name}: {message} ({timestamp}){RESET}")
                    completed_count += 1
                elif status_type == "error":
                    print(f"{ERROR}{dir_name}: {message} ({timestamp}){RESET}")
                    error_count += 1
                else:
                    print(f"{INFO}{dir_name}: {message} ({timestamp}){RESET}")
                    active_count += 1

            print("-" * 80)
            print(
                f"{INFO}Summary: {active_count} active, {completed_count} completed, {error_count} failed{RESET}"
            )
            print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
