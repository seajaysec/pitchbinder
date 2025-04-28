import argparse
import concurrent.futures
import csv
import multiprocessing
import os
import random
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
    """Convert a chord name to a shortened filename-safe version"""
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
        else:
            return flat_to_sharp.get(flat_note, flat_note), int(octave)

    return None, None


def get_note_frequency(note, octave):
    """Convert a note and octave to its frequency in Hz."""
    # Define base frequencies for A4
    A4_FREQ = 440.0
    A4_NOTE = "A"
    A4_OCTAVE = 4

    # Map note to semitone offset from C (C=0, C#=1, D=2, etc.)
    NOTE_TO_SEMITONE = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }

    # Calculate semitone distance from A4
    semitone_offset_from_A4 = (octave - A4_OCTAVE) * 12 + (
        NOTE_TO_SEMITONE[note] - NOTE_TO_SEMITONE[A4_NOTE]
    )

    # Calculate frequency using equal temperament formula
    frequency = A4_FREQ * (2 ** (semitone_offset_from_A4 / 12))
    return frequency


def get_all_wav_files(directory="."):
    """Get all WAV files in a directory and its subdirectories."""
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".wav"):
                if root == directory:
                    wav_files.append(file)
                else:
                    # Include the subdirectory in the path
                    subdir = os.path.relpath(root, directory)
                    wav_files.append(os.path.join(subdir, file))
    return wav_files


def pitch_shift_sample(
    audio_data, sr, source_note, source_octave, target_note, target_octave
):
    """Pitch shift a sample from source note to target note using resampling."""
    # Calculate the frequency ratio
    source_freq = get_note_frequency(source_note, source_octave)
    target_freq = get_note_frequency(target_note, target_octave)
    ratio = target_freq / source_freq

    # For small shifts, use librosa's implementation
    if 0.5 <= ratio <= 2.0:
        n_steps = 12 * np.log2(ratio)
        return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=n_steps), sr
    else:
        # For larger shifts, use resampling
        new_length = int(len(audio_data) / ratio)
        return scipy_signal.resample(audio_data, new_length), sr


def find_closest_sample(target_note, target_octave, existing_samples):
    """Find the closest sample to the target note for pitch shifting."""
    # Parse note and octave info from filenames
    samples_info = []
    for sample in existing_samples:
        note, octave = parse_note_from_filename(sample)
        if note and octave is not None:
            samples_info.append((sample, note, octave))

    if not samples_info:
        print_warning("  No valid samples found with note and octave information")
        return None, None, None

    # Calculate frequency distances
    target_freq = get_note_frequency(target_note, target_octave)
    distances = []

    for sample, note, octave in samples_info:
        source_freq = get_note_frequency(note, octave)
        # Use a logarithmic distance metric for more musical shifts
        if source_freq > 0 and target_freq > 0:
            # Use ratio of frequencies
            ratio = target_freq / source_freq
            if ratio < 1:
                ratio = 1 / ratio  # Make sure ratio >= 1
            distance = ratio - 1  # 0 = perfect match
            distances.append((distance, sample, note, octave))

    if not distances:
        print_warning("  Failed to calculate frequency distances")
        return None, None, None

    # Find the best match
    best_match = min(distances, key=lambda x: x[0])
    distance, sample, note, octave = best_match

    # Warn about potentially extreme shifts
    if distance > 3.0:  # More than 3 octaves
        print_warning(
            f"  Warning: closest available sample is {distance:.1f} octaves away from target"
        )
    elif distance > 1.0:  # More than 1 octave
        print_info(
            f"  Note: closest available sample is {distance:.1f} octaves away from target"
        )

    return sample, note, octave


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
    pitch_shift_method="standard",  # Keep parameter but don't use it
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
    pitch_shift_method="standard",  # Parameter for pitch shift method
):
    """Generate chord samples based on the provided chord definitions."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Use the hardcoded chord definitions
    chord_defs = CHORD_DEFINITIONS

    # Flag to track if any chords were generated
    chords_generated = False

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
        print_info("Root position chords will always be generated.")
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
            root_chords_generated = (
                False  # Track if we've generated any root position chords
            )
            for octave in range(2, 5):
                for note in notes:
                    # Skip if the highest note in the chord would be above B8
                    highest_semitone = max(semitones)
                    highest_note, highest_octave = get_note_from_semitone(
                        note, octave, highest_semitone
                    )
                    if highest_octave > 8:
                        continue

                    # Generate the root position chord regardless of pitch_shift_method
                    chord_audio, sr = generate_chord(
                        note,
                        octave,
                        semitones,
                        all_samples,
                        source_dir,
                        chord_dir,
                        prefix,
                        1.0,  # Use lower duration factor for core chords
                        pitch_shift_method,
                    )

                    if chord_audio is not None:
                        # Save the chord
                        chord_filename = (
                            f"{prefix}-{safe_chord_name}-{note}{octave}.wav"
                        )
                        chord_path = os.path.join(quality_dir, chord_filename)
                        sf.write(chord_path, chord_audio, sr)
                        tqdm.write(f"{SUCCESS}    Generated {chord_filename}{RESET}")
                        root_chords_generated = True
                        chords_generated = True

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
                                        1.0,  # No duration factor for inversion
                                        pitch_shift_method,
                                    )

                                    # Check if inversion generation returned None
                                    if inv_chord_audio is None:
                                        tqdm.write(
                                            f"{WARNING}    Inversion generation returned None for inversion {inv_num} of {chord_name} at {new_root}{new_octave}{RESET}"
                                        )
                                        raise ValueError("Inversion generation failed")

                                    # Save the inverted chord with inversion number in filename
                                    inv_chord_filename = f"{prefix}-{safe_chord_name}-{inv_num}stInv-{new_root}{new_octave}.wav"
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

            # Now generate the extended range (C1-B1 and C5-B8) by pitch shifting core chords
            if core_chords:  # Only proceed if core chords were generated
                tqdm.write(
                    f"{INFO}    Generating extended octaves (1, 5-8) for {chord_name} by pitch shifting...{RESET}"
                )
                for octave in range(1, 9):
                    # Skip the core octaves we already generated directly
                    if 2 <= octave <= 4:
                        continue

                    for note in notes:
                        # Skip if the highest note in the chord would be above B8
                        highest_semitone = max(semitones)
                        highest_note_check, highest_octave_check = (
                            get_note_from_semitone(note, octave, highest_semitone)
                        )
                        if highest_octave_check > 8:
                            continue

                        # Find the closest core chord to use as source
                        closest_core = None
                        min_distance = float("inf")

                        for core_note, core_octave in core_chords.keys():
                            core_index = notes.index(core_note) + (core_octave * 12)
                            target_index = notes.index(note) + (octave * 12)
                            distance = abs(target_index - core_index)

                            if distance < min_distance:
                                min_distance = distance
                                closest_core = (core_note, core_octave)

                        if closest_core:
                            source_note, source_octave = closest_core
                            # Make sure key exists before accessing
                            if closest_core in core_chords:
                                _, chord_audio, sr = core_chords[closest_core]
                            else:
                                tqdm.write(
                                    f"{WARNING}    Core chord {closest_core} not found for pitch shifting {note}{octave}{RESET}"
                                )
                                continue  # Skip if source chord data isn't available

                            # Pitch shift the chord audio
                            new_audio, new_sr = pitch_shift_sample(
                                chord_audio,
                                sr,
                                source_note,
                                source_octave,
                                note,
                                octave,
                            )

                            # Time stretch to maintain duration (based on source core chord)
                            source_duration = len(chord_audio) / sr
                            current_duration = len(new_audio) / new_sr
                            # Only stretch if duration differs significantly (e.g., more than 1%)
                            if (
                                abs(source_duration - current_duration)
                                > 0.01 * source_duration
                            ):
                                stretch_factor = source_duration / current_duration
                                rate = (
                                    1.0 / stretch_factor
                                )  # Rate for librosa.effects.time_stretch

                                # Ensure audio is float64 numpy array
                                if isinstance(new_audio, tuple):
                                    new_audio_array = new_audio[0].astype(np.float64)
                                else:
                                    new_audio_array = new_audio.astype(np.float64)

                                # Adjust n_fft for short samples
                                n_fft = 2048
                                if len(new_audio_array) < n_fft:
                                    fft_power = (
                                        np.log2(len(new_audio_array) - 1)
                                        if len(new_audio_array) > 1
                                        else 5
                                    )
                                    n_fft = 2 ** int(fft_power)
                                    n_fft = max(32, n_fft)  # Ensure minimum n_fft

                                try:
                                    new_audio = librosa.effects.time_stretch(
                                        new_audio_array, rate=float(rate), n_fft=n_fft
                                    )

                                    # Apply fade out
                                    fade_len = min(
                                        int(new_sr * 0.1), len(new_audio) // 10
                                    )
                                    if fade_len > 0:
                                        envelope = np.ones(len(new_audio))
                                        envelope[-fade_len:] = np.linspace(
                                            1, 0, fade_len
                                        )
                                        new_audio *= envelope
                                except Exception as stretch_e:
                                    tqdm.write(
                                        f"{WARNING}    Time stretching failed for {note}{octave}: {str(stretch_e)}. Using original pitch-shifted audio.{RESET}"
                                    )
                                    # Use new_audio without stretching if it fails

                                # Normalize
                                if np.max(np.abs(new_audio)) > 0:
                                    new_audio = (
                                        new_audio / np.max(np.abs(new_audio)) * 0.95
                                    )

                                # Save the pitch-shifted chord
                                shifted_chord_filename = (
                                    f"{prefix}-{safe_chord_name}-{note}{octave}.wav"
                                )
                                shifted_chord_path = os.path.join(
                                    quality_dir, shifted_chord_filename
                                )
                                sf.write(shifted_chord_path, new_audio, new_sr)
                                tqdm.write(
                                    f"{SUCCESS}    Generated {shifted_chord_filename} (pitch-shifted){RESET}"
                                )
                                chords_generated = (
                                    True  # Mark that we generated something
                                )
                                master_pbar.update(1)  # Update master progress

                                # Generate inversions for this pitch-shifted chord if requested
                                if generate_inversions and inversions:
                                    for inv_num, inv_semitones in inversions:
                                        # Skip if this inversion is not selected
                                        if selected_inversions is not None:
                                            chord_key = (quality, chord_name)
                                            if (
                                                chord_key not in selected_inversions
                                                or inv_num
                                                not in selected_inversions[chord_key]
                                            ):
                                                continue

                                        # Compute the new root for the inversion based on the shifted root note/octave
                                        new_root, new_octave = get_note_from_semitone(
                                            note,
                                            octave,
                                            semitones[
                                                inv_num
                                            ],  # Use the SHIFTED note/octave as base
                                        )

                                        # Save the pitch-shifted audio data with the correct inversion filename
                                        inv_chord_filename = f"{prefix}-{safe_chord_name}-{inv_num}stInv-{new_root}{new_octave}.wav"
                                        inv_chord_path = os.path.join(
                                            inversions_dir, inv_chord_filename
                                        )

                                        # Re-use the already pitch-shifted and time-stretched audio
                                        sf.write(inv_chord_path, new_audio, new_sr)
                                        tqdm.write(
                                            f"{SUCCESS}    Generated {inv_chord_filename} (pitch-shifted){RESET}"
                                        )
                                        master_pbar.update(1)  # Update master progress
                        else:
                            tqdm.write(
                                f"{WARNING}    Could not find a suitable core chord to pitch shift for {note}{octave}{RESET}"
                            )
            else:
                tqdm.write(
                    f"{WARNING}    Skipping extended octaves for {chord_name} because no core chords were generated.{RESET}"
                )

            # Update chord progress bar for the current chord type
            with tqdm_lock:
                chord_pbar.update(1)

    # Close progress bars
    with tqdm_lock:
        quality_pbar.close()
        master_pbar.close()

    # Check if we generated any chords at all
    if not chords_generated:
        print_warning("No chords were generated. Check your sample files and settings.")
        print_info(
            "Make sure your source directory contains valid note samples (e.g., C4, G#3)."
        )
        print_info(
            "Also check that you selected appropriate chord qualities and pitch shift method."
        )

    return chord_dir, []


def generate_missing_samples(
    prefix,
    existing_samples,
    source_dir,
    target_dir,
    time_match=False,
    pitch_shift_method="standard",
    keep_artifacts=False,
):
    """Generate all missing samples across the 8-octave range."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print_info(f"Generating missing samples for prefix: {prefix}")

    # Parse existing samples
    existing_notes = {}
    bad_samples = []
    durations = []  # store durations for time matching

    for sample in existing_samples:
        note, octave = parse_note_from_filename(sample)
        if note and octave:
            existing_notes[(note, octave)] = sample

            # If we're time matching, get the durations
            if time_match:
                try:
                    audio_path = os.path.join(source_dir, sample)
                    audio_data, sr = librosa.load(audio_path, sr=None)
                    duration = len(audio_data) / sr
                    durations.append(duration)
                except Exception as e:
                    print_warning(f"  Failed to get duration for {sample}: {str(e)}")
        else:
            bad_samples.append(sample)

    if bad_samples:
        print_warning(
            f"  {len(bad_samples)} samples couldn't be parsed for note/octave and will be ignored:"
        )
        for sample in bad_samples[:5]:  # Show just a few examples
            print_warning(f"    - {sample}")
        if len(bad_samples) > 5:
            print_warning(f"    - (and {len(bad_samples) - 5} more)")

    # Calculate average duration for time matching
    avg_duration = None
    if time_match and durations:
        avg_duration = sum(durations) / len(durations)
        print_info(f"  Average sample duration: {avg_duration:.2f}s")

    # Generate missing notes for all octaves (C0 to B8)
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octaves = range(0, 9)  # 0-8
    generated_files = []

    # Progress bar
    total_notes = len(notes) * len(octaves)
    existing_count = len(existing_notes)
    missing_count = total_notes - existing_count

    print_info(
        f"  Found {existing_count} existing notes, generating {missing_count} missing notes"
    )

    with tqdm(total=missing_count, desc="Generating notes", position=0) as progress_bar:
        for octave in octaves:
            for note in notes:
                target_note = note
                target_octave = octave
                target_filename = f"{prefix}-{target_note}{target_octave}.wav"

                # Skip if file already exists
                if (target_note, target_octave) in existing_notes:
                    continue

                # Find the closest sample
                closest_file, closest_note, closest_octave = find_closest_sample(
                    target_note, target_octave, existing_samples
                )

                # Skip if no close sample found
                if not closest_file:
                    with tqdm_lock:
                        print_warning(
                            f"  No suitable source sample found for {target_note}{target_octave}, skipping"
                        )
                    continue

                # Load the source audio
                closest_path = os.path.join(source_dir, closest_file)
                try:
                    audio_data, sr = librosa.load(closest_path, sr=None)
                except Exception as e:
                    with tqdm_lock:
                        print_error(
                            f"  Failed to load {closest_file}: {str(e)}, skipping {target_note}{target_octave}"
                        )
                    continue

                # Standard pitch shifting
                new_audio, new_sr = pitch_shift_sample(
                    audio_data,
                    sr,
                    closest_note,
                    closest_octave,
                    target_note,
                    target_octave,
                )

                # Time stretch if requested
                if time_match and durations and avg_duration is not None:
                    current_duration = len(new_audio) / new_sr
                    stretch_factor = avg_duration / current_duration
                    if abs(stretch_factor - 1.0) > 0.01:
                        with tqdm_lock:
                            print_info(
                                f"  Time stretching (target: {avg_duration:.2f}s, factor: {stretch_factor:.2f})"
                            )
                        # Use librosa's time stretch
                        # Ensure new_audio is a numpy array before passing to time_stretch
                        if isinstance(new_audio, tuple):
                            new_audio_array = new_audio[0]
                        else:
                            new_audio_array = new_audio

                        new_audio = librosa.effects.time_stretch(
                            new_audio_array, rate=1.0 / stretch_factor
                        )

                # Normalize output
                if np.max(np.abs(new_audio)) > 0:
                    new_audio = new_audio / np.max(np.abs(new_audio)) * 0.95

                # Save the standard method output
                output_path = os.path.join(target_dir, target_filename)
                sf.write(output_path, new_audio, new_sr)

                with tqdm_lock:
                    print_success(f"Generated {target_filename} using standard method")
                generated_files.append(target_filename)

                # Update progress bar
                with tqdm_lock:
                    progress_bar.update(1)

    # Return list of all samples (existing + generated)
    all_samples = existing_samples.copy()

    # Add generated files with proper paths for expansion directory
    for gen_file in generated_files:
        all_samples.append(gen_file)

    return all_samples


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


def cleanup_artifacts(
    source_dir,
    target_dir,
    chord_dir,
    full_sample_filename,
    full_chord_filenames,
):
    """Clean up artifact files after processing, keeping only the full sample files."""
    print_header("Cleaning up artifacts")

    # Create the exp directory if it doesn't exist
    exp_dir = os.path.join(source_dir, "exp")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # First, check if the full sample file exists in the exp directory
    # If not, it might be in the target directory
    if full_sample_filename:
        full_sample_path_exp = os.path.join(exp_dir, full_sample_filename)
        full_sample_path_target = os.path.join(target_dir, full_sample_filename)

        # If the file doesn't exist in exp but exists in target or source, copy it
        if not os.path.exists(full_sample_path_exp):
            import shutil

            if os.path.exists(full_sample_path_target):
                shutil.copy2(full_sample_path_target, full_sample_path_exp)
                print_success(f"Copied full sample file to {full_sample_path_exp}")
            else:
                print_warning(
                    f"Full sample file {full_sample_filename} not found in target or source directories."
                )

    # Now remove the temporary directories
    import shutil

    def robust_rmtree(directory):
        """Attempt to remove a directory more aggressively if standard rmtree fails."""
        try:
            shutil.rmtree(directory)
            return True
        except OSError as e:
            print_warning(f"Initial removal of {directory} failed: {e}")
            print_info("Attempting more aggressive cleanup...")

            try:
                # Manually remove files first
                for root, dirs, files in os.walk(directory, topdown=False):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            os.unlink(file_path)
                            print_info(f"Removed file: {file_path}")
                        except Exception as e:
                            print_warning(f"Could not remove file {file_path}: {e}")

                    # Then try to remove directories
                    for dir in dirs:
                        try:
                            dir_path = os.path.join(root, dir)
                            os.rmdir(dir_path)
                            print_info(f"Removed directory: {dir_path}")
                        except Exception as e:
                            print_warning(f"Could not remove directory {dir_path}: {e}")

                # Try to remove the main directory again
                os.rmdir(directory)
                print_success(
                    f"Successfully removed {directory} after aggressive cleanup"
                )
                return True
            except OSError as e:
                print_warning(
                    f"Could not completely remove {directory} even after aggressive cleanup: {e}"
                )
                print_info("Continuing anyway as this won't affect the results.")
                return False

    # Remove the temporary expansion directory
    if os.path.exists(target_dir) and target_dir != exp_dir:
        print_info(f"Removing expansion directory: {target_dir}")
        robust_rmtree(target_dir)

    # Remove the temporary chord directory
    if (
        chord_dir
        and os.path.exists(chord_dir)
        and chord_dir != os.path.join(exp_dir, "chords")
    ):
        print_info(f"Removing chord directory: {chord_dir}")
        robust_rmtree(chord_dir)

    print_success("Cleanup complete")


def process_directory(
    source_dir,
    target_dir,
    prefix=None,
    play=False,
    gen_full=False,
    time_match=False,
    chords=False,
    keep_artifacts=False,
    chord_qualities=None,
    generate_inversions=False,
    selected_chord_types=None,  # New parameter for selected chord types
    selected_inversions=None,  # New parameter for selected inversions
    overwrite=False,  # Parameter to control overwriting existing directories
    pitch_shift_method="standard",  # Parameter to control pitch shift method: "standard" only
):
    """Process a single directory to generate missing samples."""
    # Acquire lock for consistent console output when running in parallel
    with tqdm_lock:
        print_header(f"Processing directory: {source_dir}")

        # If the directory contains a subdirectory named 'exp' and overwrite is not enabled, skip processing.
        exp_dir = os.path.join(source_dir, "exp")
        if not overwrite and os.path.isdir(exp_dir):
            print_info(
                f"Skipping directory: {source_dir} because it contains an 'exp' subdirectory"
            )
            update_status(
                source_dir, "Directory skipped due to 'exp' subdirectory", "warning"
            )
            return False

        # Get all WAV files in the source directory
        existing_samples = get_all_wav_files(source_dir)

        if not existing_samples:
            update_status(
                source_dir, "No WAV files found in the source directory", "warning"
            )
            return False

        # Check if files have detectable notes
        valid_samples = []
        for sample in existing_samples:
            note, octave = parse_note_from_filename(sample)
            if note and octave:
                valid_samples.append(sample)

        if not valid_samples:
            update_status(
                source_dir, "No samples with detectable notes found", "warning"
            )
            return False

        if len(valid_samples) < len(existing_samples):
            update_status(
                source_dir,
                f"{len(existing_samples) - len(valid_samples)} samples have undetectable notes and will be ignored",
                "warning",
            )

        # Auto-detect prefix if not provided
        dir_prefix = prefix
        if not dir_prefix:
            # Use the prefix from the first valid sample
            match = re.match(r"(.+)-[A-G]#?\d+\.wav", valid_samples[0])
            if match:
                # Extract prefix, but only use the part after the last directory separator
                full_prefix = match.group(1)
                dir_prefix = os.path.basename(full_prefix)
            else:
                # Use the directory name as a fallback
                dir_prefix = os.path.basename(source_dir)
                if not dir_prefix:  # In case it's the root directory
                    dir_prefix = "Piano"

        # Now that we've validated the samples, create the target directory
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        update_status(source_dir, f"Using prefix: {dir_prefix}", "info")
        update_status(source_dir, f"Found {len(valid_samples)} valid samples", "info")
        update_status(
            source_dir, f"Generated samples will be saved to {target_dir}", "info"
        )

    # Release lock during time-consuming operations for better parallelism

    # Generate missing samples
    all_samples = []
    if valid_samples:
        print_header(f"Generating missing notes for {os.path.basename(source_dir)}")
        update_status(source_dir, "Generating missing samples...", "info")
        all_samples = generate_missing_samples(
            dir_prefix,
            valid_samples,
            source_dir,
            target_dir,
            time_match,
            pitch_shift_method,
            keep_artifacts,
        )

    # Use lock for progress output
    update_status(source_dir, f"Generated {len(all_samples)} total samples", "success")

    # Generate chord samples if requested
    chord_dir = None
    full_chord_filenames = []
    if chords:
        update_status(source_dir, "Starting chord generation...", "info")
        chord_dir = os.path.join(source_dir, "exp_chords")
        generated_chord_dir, full_chord_filenames = generate_chords(
            dir_prefix,
            all_samples,
            source_dir,
            chord_dir,
            target_dir,  # Pass the expansion directory
            chord_qualities=chord_qualities,
            generate_inversions=generate_inversions,
            selected_chord_types=selected_chord_types,  # Pass selected chord types
            selected_inversions=selected_inversions,  # Pass selected inversions
            pitch_shift_method=pitch_shift_method,  # Pass pitch shift method
        )
        # Keep the original chord_dir value for the cleanup step
        update_status(
            source_dir,
            f"Chord generation complete: {len(full_chord_filenames)} chord types created",
            "success",
        )

    # Generate full sample file if requested
    full_sample_filename = None
    if gen_full:
        update_status(source_dir, "Generating full sample file...", "info")
        full_sample_filename = generate_full_sample(
            all_samples, dir_prefix, source_dir, target_dir
        )
        update_status(source_dir, "Full sample file generated", "success")

    # Play all notes if requested (excluding the full sample) - only when not in parallel mode
    if play:
        update_status(source_dir, "Playing all generated samples...", "info")
        # Filter out the full sample if it was generated
        samples_to_play = [s for s in all_samples if s != full_sample_filename]
        play_all_notes(samples_to_play, source_dir, target_dir)

    # Clean up artifacts if requested
    if not keep_artifacts:
        update_status(source_dir, "Cleaning up temporary files...", "info")
        cleanup_artifacts(
            source_dir,
            target_dir,
            chord_dir,
            full_sample_filename,
            full_chord_filenames,
        )
        update_status(source_dir, "Cleanup complete", "info")

    # Return success indicator
    return True


def generate_full_chord_samples(chord_dir, prefix):
    """Generate separate full sample files for each chord type with embedded slice markers."""
    tqdm.write(f"\n{INFO}{'='*60}{RESET}")
    tqdm.write(f"{INFO}Generating full chord sample files by type{RESET}")
    tqdm.write(f"{INFO}{'='*60}{RESET}")

    # Define the final output directory structure within the source directory's 'exp' folder
    source_parent_dir = os.path.dirname(
        chord_dir
    )  # Should be the instrument's source dir
    final_exp_dir = os.path.join(source_parent_dir, "exp")
    final_chords_output_dir = os.path.join(final_exp_dir, "chords")

    # Import necessary modules for embedding slice markers
    import struct
    import wave

    # Find all chord WAV files in the chord directory and its subdirectories
    chord_files = []
    for root, _, files in os.walk(chord_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                chord_files.append(os.path.join(root, file))

    if not chord_files:
        tqdm.write(f"{WARNING}No chord files found to create full samples{RESET}")
        return []

    # Group chord files by chord type
    chord_types = {}
    inversion_types = {}

    for chord_file in chord_files:
        filename = os.path.basename(chord_file)
        dir_name = os.path.basename(os.path.dirname(chord_file))

        # Check if this is an inversion
        is_inversion = "inv" in os.path.dirname(
            chord_file
        ) or "inversions" in os.path.dirname(chord_file)

        # If it's in an inversions directory, get the parent quality directory
        if is_inversion:
            quality = os.path.basename(os.path.dirname(os.path.dirname(chord_file)))
        else:
            quality = dir_name

        # Extract note and chord type
        note_match = re.search(r"([A-G]#?\d+)\.wav$", filename)
        if not note_match:
            continue

        note_str = note_match.group(1)

        # Extract chord type and inversion info
        if is_inversion:
            # Pattern for inversions: prefix-ChordType-InversionNum-NoteOctave.wav
            chord_match = re.search(
                rf"{prefix}-(.+)-(\d+stInv)-{note_str}\.wav$", filename
            )
            if not chord_match:
                continue

            chord_type = chord_match.group(1)
            inversion_num = chord_match.group(2)

            # Create a key for this inversion type
            key = (quality, chord_type, inversion_num)

            if key not in inversion_types:
                inversion_types[key] = []

            inversion_types[key].append(chord_file)
        else:
            # Regular chord pattern: prefix-ChordType-NoteOctave.wav
            chord_match = re.search(rf"{prefix}-(.+)-{note_str}\.wav$", filename)
            if not chord_match:
                continue

            chord_type = chord_match.group(1)

            # Create a key for this chord type
            key = (quality, chord_type)

            if key not in chord_types:
                chord_types[key] = []

            chord_types[key].append(chord_file)

    total_types = len(chord_types) + len(inversion_types)
    tqdm.write(
        f"{INFO}Found {len(chord_types)} chord types and {len(inversion_types)} inversion types to process{RESET}"
    )

    # Create a progress bar for chord types
    pbar = tqdm(
        total=total_types,
        desc="Creating full chord samples",
        position=0,
        leave=True,
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, RESET),
    )

    # Process each chord type
    full_chord_filenames = []

    # Process regular chords
    for (quality, chord_type), files in chord_types.items():
        # Sort files by note and octave
        def sort_key(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r"([A-G]#?)(\d+)\.wav$", filename)
            if match:
                note, octave = match.groups()
                notes = [
                    "C",
                    "C#",
                    "D",
                    "D#",
                    "E",
                    "F",
                    "F#",
                    "G",
                    "G#",
                    "A",
                    "A#",
                    "B",
                ]
                note_index = notes.index(note) if note in notes else 0
                octave_num = int(octave) if octave.isdigit() else 0
                return (octave_num, note_index)
            return (0, 0)

        sorted_files = sorted(files, key=sort_key)

        # Skip if no files found
        if not sorted_files:
            tqdm.write(f"{WARNING}No files found for {quality} {chord_type}{RESET}")
            continue

        # Load the first file to get sample rate
        first_audio, sr = librosa.load(sorted_files[0], sr=None)

        # Calculate silence duration (0.5 seconds)
        silence_duration = int(0.5 * sr)
        silence = np.zeros(silence_duration)

        # Prepare for combined audio
        combined_audio = np.array([])

        # Prepare for slice markers
        cue_positions = []  # Store (label, position) tuples
        current_position = 0  # in samples

        # Process each chord file
        for i, chord_file in enumerate(sorted_files):
            # Extract note info for slice marker
            filename = os.path.basename(chord_file)
            note_match = re.search(r"([A-G]#?\d+)\.wav$", filename)
            note_str = note_match.group(1) if note_match else "Unknown"

            # Load the audio
            audio, _ = librosa.load(chord_file, sr=sr)

            # Trim silence at the beginning and end
            # Ensure audio is a numpy array
            if isinstance(audio, tuple):
                audio = audio[0]

            # Convert to float64 for librosa.effects.trim
            audio_float = (
                audio.astype(np.float64) if hasattr(audio, "astype") else audio
            )
            audio, _ = librosa.effects.trim(audio_float, top_db=30)

            # Limit each sample to 3 seconds max
            max_length = min(len(audio), 3 * sr)
            audio = audio[:max_length]

            # Add a fade out
            fade_samples = min(
                int(0.1 * sr), len(audio) // 4
            )  # 100ms fade or 1/4 of length
            if fade_samples > 0:
                audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

            # Record the start position of this chord (in samples)
            cue_positions.append((f"{note_str}", current_position))

            # Add the chord sample to the combined audio
            combined_audio = np.concatenate([combined_audio, audio])

            # Update current position for next slice marker
            current_position += len(audio)

            # Add silence between samples (but not after the last one)
            if i < len(sorted_files) - 1:
                combined_audio = np.concatenate([combined_audio, silence])
                current_position += silence_duration

        # Create a safe filename from the chord type
        safe_chord_type = chord_to_filename(chord_type)

        # Define the final output directory and ensure it exists
        final_quality_dir = os.path.join(final_chords_output_dir, quality)
        if not os.path.exists(final_quality_dir):
            os.makedirs(final_quality_dir)

        # Save the combined audio with embedded slice markers
        output_filename = f"{safe_chord_type}.wav"  # Removed prefix and note
        output_path = os.path.join(final_quality_dir, output_filename)

        # First save the audio data using soundfile
        sf.write(output_path, combined_audio, sr)
        tqdm.write(f"Generated final chord file: {output_path}")

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
            tqdm.write(
                f"{SUCCESS}Added {len(cue_positions)} slice markers to {output_filename}{RESET}"
            )

        except Exception as e:
            tqdm.write(f"{WARNING}Could not add slice markers to WAV file: {e}{RESET}")

        # Store the quality and filename for later use
        full_chord_filenames.append((quality, output_filename))

        tqdm.write(
            f"{SUCCESS}Generated full sample for {quality} {chord_type}: {output_filename}{RESET}"
        )

        pbar.update(1)

    # Process inversion types
    for (quality, chord_type, inversion_num), files in inversion_types.items():
        # Sort files by note and octave using the same sort function as before
        sorted_files = sorted(files, key=sort_key)

        # Skip if no files found
        if not sorted_files:
            tqdm.write(
                f"{WARNING}No inversion files found for {quality} {chord_type} {inversion_num}{RESET}"
            )
            continue

        # Load the first file to get sample rate
        first_audio, sr = librosa.load(sorted_files[0], sr=None)

        # Calculate silence duration (0.5 seconds)
        silence_duration = int(0.5 * sr)
        silence = np.zeros(silence_duration)

        # Prepare for combined audio
        combined_audio = np.array([])

        # Prepare for slice markers
        cue_positions = []  # Store (label, position) tuples
        current_position = 0  # in samples

        # Process each inversion file
        for i, inversion_file in enumerate(sorted_files):
            # Extract note info for slice marker
            filename = os.path.basename(inversion_file)
            note_match = re.search(r"([A-G]#?\d+)\.wav$", filename)
            note_str = note_match.group(1) if note_match else "Unknown"

            # Load the audio
            audio, _ = librosa.load(inversion_file, sr=sr)

            # Trim silence at the beginning and end
            # Ensure audio is a numpy array
            if isinstance(audio, tuple):
                audio = audio[0]

            # Convert to float64 for librosa.effects.trim
            audio_float = (
                audio.astype(np.float64) if hasattr(audio, "astype") else audio
            )
            audio, _ = librosa.effects.trim(audio_float, top_db=30)

            # Limit each sample to 3 seconds max
            max_length = min(len(audio), 3 * sr)
            audio = audio[:max_length]

            # Add a fade out
            fade_samples = min(
                int(0.1 * sr), len(audio) // 4
            )  # 100ms fade or 1/4 of length
            if fade_samples > 0:
                audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

            # Record the start position of this inversion (in samples)
            cue_positions.append((f"{note_str}", current_position))

            # Add the inversion sample to the combined audio
            combined_audio = np.concatenate([combined_audio, audio])

            # Update current position for next slice marker
            current_position += len(audio)

            # Add silence between samples (but not after the last one)
            if i < len(sorted_files) - 1:
                combined_audio = np.concatenate([combined_audio, silence])
                current_position += silence_duration

        # Create a safe filename from the chord type and inversion number
        safe_chord_type = chord_to_filename(chord_type)

        # Define final output directory for inversions and ensure it exists
        final_quality_dir = os.path.join(final_chords_output_dir, quality)
        final_inversions_dir = os.path.join(final_quality_dir, "inv")
        if not os.path.exists(final_inversions_dir):
            os.makedirs(final_inversions_dir)

        # Save the combined audio with embedded slice markers
        output_filename = (
            f"{safe_chord_type}-{inversion_num}.wav"  # Removed prefix and note
        )
        output_path = os.path.join(final_inversions_dir, output_filename)

        # First save the audio data using soundfile
        sf.write(output_path, combined_audio, sr)
        tqdm.write(f"Generated final inversion file: {output_path}")

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
                with open(output_path + ".temp", "ab") as f:
                    f.write(b"cue ")  # Chunk ID
                    f.write(struct.pack("<I", len(cue_chunk)))  # Chunk size
                    f.write(cue_chunk)  # Chunk data

            # Replace the original file with the new one
            os.replace(output_path + ".temp", output_path)
            tqdm.write(
                f"{SUCCESS}Added {len(cue_positions)} slice markers to {output_filename}{RESET}"
            )

        except Exception as e:
            tqdm.write(f"{WARNING}Could not add slice markers to WAV file: {e}{RESET}")

        # Store the quality, subdir, and filename for later use
        full_chord_filenames.append((quality, "inv", output_filename))

        tqdm.write(
            f"{SUCCESS}Generated full sample for {quality} {chord_type} {inversion_num}: {output_filename}{RESET}"
        )

        pbar.update(1)

    pbar.close()

    tqdm.write(
        f"{SUCCESS}Generated {len(full_chord_filenames)} full chord sample files{RESET}"
    )
    return full_chord_filenames


def interactive_mode():
    """Run the tool in interactive mode using questionary."""
    print(f"\n{HIGHLIGHT}{'='*60}")
    print(f"{HIGHLIGHT}{'='*15} NOTE SAMPLE EXPANDER {'='*15}")
    print(f"{HIGHLIGHT}{'='*60}{RESET}\n")

    print_info(
        "Welcome to the note-to-octave sample expander - now with chord generation! Let's get set up."
    )

    # Create a custom path completer that handles path expansion
    def path_completer(text, document, complete_event):
        # Expand user paths like ~ and environment variables
        expanded_text = os.path.expanduser(os.path.expandvars(text))

        # Get the directory to look in
        if os.path.isdir(expanded_text):
            directory = expanded_text
            prefix = ""
        else:
            directory = os.path.dirname(expanded_text) or "."
            prefix = os.path.basename(expanded_text)

        # List directories and files
        try:
            names = os.listdir(directory)
            # Filter based on prefix and only include directories
            names = [
                name + "/" if os.path.isdir(os.path.join(directory, name)) else name
                for name in names
                if name.startswith(prefix)
            ]
            # Add the directory part back
            if directory != ".":
                names = [os.path.join(directory, name) for name in names]
            return names
        except (PermissionError, FileNotFoundError):
            return []

    # Use questionary with custom completer
    from prompt_toolkit.completion import Completer, Completion

    class PathCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text
            completions = path_completer(text, document, complete_event)
            for completion in completions:
                yield Completion(
                    completion,
                    start_position=-len(document.text),
                )

    # Get source directory with autocomplete
    source_dir_input = questionary.text(
        "Enter the source directory containing WAV samples:",
        default=os.getcwd(),  # Use current working directory as default
        style=custom_style,
        completer=PathCompleter(),
    ).ask()

    # Expand user paths like ~ and environment variables
    source_dir = os.path.expanduser(os.path.expandvars(source_dir_input))
    source_dir = os.path.abspath(source_dir)  # Convert to absolute path

    # Check if directory exists and has WAV files
    if not os.path.isdir(source_dir):
        print_error(f"Directory '{source_dir}' does not exist!")
        # Ask if user wants to try again
        if questionary.confirm(
            "Would you like to try a different directory?", style=custom_style
        ).ask():
            return interactive_mode()  # Restart the interactive mode
        return

    # Verify the directory is accessible
    try:
        wav_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".wav")]
        print_info(f"Found {len(wav_files)} WAV files in {source_dir}")
    except PermissionError:
        print_error(f"Permission denied when accessing {source_dir}")
        if questionary.confirm(
            "Would you like to try a different directory?", style=custom_style
        ).ask():
            return interactive_mode()
        return
    except Exception as e:
        print_error(f"Error accessing directory: {e}")
        if questionary.confirm(
            "Would you like to try a different directory?", style=custom_style
        ).ask():
            return interactive_mode()
        return

    if not wav_files:
        print_warning(
            f"No WAV files found in {source_dir}. Will look in subdirectories if any exist."
        )

    # Always use recursive mode
    recurse = True
    print_info("Using recursive mode to process all subdirectories")

    # Set parallel processing to True by default without asking
    use_parallel = True
    # Calculate optimal number of workers based on system resources
    max_workers = get_optimal_workers()
    print_info(f"Using {max_workers} parallel workers for processing")

    # Initialize prefix to None
    prefix = None

    # Ask about other options
    options = questionary.checkbox(
        "Select additional options:",
        choices=[
            questionary.Choice(
                "Generate a single WAV file with all notes in sequence",
                "gen_full",
                checked=True,
            ),
            questionary.Choice(
                "Match all generated samples to the average length of source samples",
                "time_match",
                checked=True,
            ),
            questionary.Choice("Generate chord samples", "chords", checked=True),
            questionary.Choice("Play all notes when done", "play"),
            questionary.Choice("Overwrite existing expansion directories", "overwrite"),
            questionary.Choice(
                "Keep all generated files (don't clean up artifacts)", "keep_artifacts"
            ),
            questionary.Choice(
                "Use a custom prefix for generated files (Otherwise auto-detect)",
                "use_custom_prefix",
            ),
            questionary.Choice(
                "Set custom number of parallel workers", "set_custom_workers"
            ),
        ],
        style=custom_style,
    ).ask()

    # Convert options to a dictionary
    options_dict = {
        "gen_full": "gen_full" in options,
        "time_match": "time_match" in options,
        "chords": "chords" in options,
        "play": "play" in options,
        "overwrite": "overwrite" in options,
        "keep_artifacts": "keep_artifacts" in options,
        "use_custom_prefix": "use_custom_prefix" in options,
        "set_custom_workers": "set_custom_workers" in options,
    }

    # Set pitch shift method to standard (no menu selection needed)
    pitch_shift_method = "standard"

    # If the user wants to generate chords, ask additional questions about chord types and inversions
    selected_chord_types = None
    selected_inversions = None
    chord_qualities = None  # Initialize chord_qualities variable even if chords=False

    if options_dict["chords"]:
        # Ask about chord qualities
        print_info("\nThe following chord qualities are available:")

        # Extract unique chord qualities from CHORD_DEFINITIONS
        all_qualities = sorted(set(quality for _, quality, _, _ in CHORD_DEFINITIONS))
        print_info(f"Available qualities: {', '.join(all_qualities)}")

        # Create choice objects for each quality
        quality_choices = [
            questionary.Choice(quality, quality) for quality in all_qualities
        ]

        # Ask which chord qualities to generate
        selected_qualities = questionary.checkbox(
            "Which chord qualities would you like to generate?",
            choices=quality_choices,
            style=custom_style,
        ).ask()

        # If none selected, use all qualities
        chord_qualities = (
            selected_qualities
            if selected_qualities
            else sorted(set(quality for _, quality, _, _ in CHORD_DEFINITIONS))
        )

        # Only ask about specific chord types if there are chord qualities selected
        specific_types = False
        if selected_qualities:
            # Ask about specific chord types within the selected qualities
            specific_types = questionary.confirm(
                "Would you like to select specific chord types within these qualities?",
                default=False,
                style=custom_style,
            ).ask()

        if specific_types:
            # Group chord types by quality for selection
            chord_types_by_quality = {}
            for name, q, semitones, notes_count in CHORD_DEFINITIONS:
                if q in chord_qualities:
                    if q not in chord_types_by_quality:
                        chord_types_by_quality[q] = []
                    chord_types_by_quality[q].append(name)

            # Create a dictionary to store the selected types
            selected_chord_types = {}

            # For each selected quality, ask which chord types to generate
            for quality in chord_qualities:
                if quality in chord_types_by_quality:
                    # Create choice objects for each chord type
                    type_choices = [
                        questionary.Choice(chord_type, chord_type)
                        for chord_type in chord_types_by_quality[quality]
                    ]

                    # Ask which chord types to generate
                    selected_types = questionary.checkbox(
                        f"Which {quality} chord types would you like to generate?",
                        choices=type_choices,
                        style=custom_style,
                    ).ask()

                    # If types were selected, add them to the dictionary
                    if selected_types:
                        selected_chord_types[quality] = selected_types

        # Ask about inversions only if there are chord qualities selected
        generate_inversions = False
        if selected_qualities:
            print_info("\nRoot position chords will always be generated.")
            generate_inversions = questionary.confirm(
                "Would you also like to generate chord inversions?",
                default=False,
                style=custom_style,
            ).ask()

        if generate_inversions:
            # Ask if they want all inversions or specific ones
            all_inversions = questionary.confirm(
                "Generate all possible inversions for each chord?",
                default=True,
                style=custom_style,
            ).ask()

            if not all_inversions:
                # If not all inversions, ask about specific inversions
                selected_inversions = {}

                # Prepare data for processing
                chord_types_to_process = []
                if specific_types and selected_chord_types:
                    # Use the selected chord types
                    for quality, types in selected_chord_types.items():
                        for chord_type in types:
                            for name, _, _, notes_count in CHORD_DEFINITIONS:
                                if name == chord_type:
                                    chord_types_to_process.append(
                                        (quality, chord_type, notes_count)
                                    )
                                    break
                else:
                    # Use all chord types within selected qualities
                    qualities_to_process = (
                        chord_qualities
                        if chord_qualities
                        else sorted(
                            set(quality for _, quality, _, _ in CHORD_DEFINITIONS)
                        )
                    )
                    for name, quality, semitones, notes_count in CHORD_DEFINITIONS:
                        if quality in qualities_to_process:
                            chord_types_to_process.append((quality, name, notes_count))

                # Function to get proper ordinal suffix
                def get_ordinal_suffix(n):
                    if 10 <= n % 100 <= 20:
                        suffix = "th"
                    else:
                        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
                    return f"{n}{suffix}"

                # Process each chord type individually for more granular control
                for quality, chord_type, notes_count in sorted(
                    chord_types_to_process, key=lambda x: (x[0], x[1])
                ):
                    if notes_count >= 3:  # Only process chords that can have inversions
                        possible_inversions = notes_count - 1
                        inversion_choices = [
                            get_ordinal_suffix(i)
                            for i in range(1, possible_inversions + 1)
                        ]

                        header = f"Select inversions for {chord_type} ({quality}) - {notes_count}-note chord:"
                        print_info(
                            f"\nPossible inversions: {', '.join(inversion_choices)}"
                        )

                        # Create choice objects with all checked by default
                        inversion_choices_objects = [
                            questionary.Choice(inv, inv, checked=True)
                            for inv in inversion_choices
                        ]

                        selected = questionary.checkbox(
                            header,
                            choices=inversion_choices_objects,
                            style=custom_style,
                        ).ask()

                        if selected:
                            # Convert from ordinal format to inversion numbers
                            inversion_numbers = [
                                int(inv.rstrip("stndrh")) for inv in selected
                            ]
                            selected_inversions[(quality, chord_type)] = (
                                inversion_numbers
                            )
                        else:
                            # No inversions selected for this chord type
                            print_info(
                                f"No inversions selected for {chord_type} ({quality})"
                            )
        else:  # No inversions
            print_info("Only root position chords will be generated.")
    else:  # No chord generation
        generate_inversions = False
        selected_inversions = []

    # Handle custom prefix if selected
    if options_dict["use_custom_prefix"]:
        prefix = questionary.text(
            "Enter the prefix for generated files:", style=custom_style
        ).ask()

    # Handle custom workers if selected and parallel processing is enabled
    if options_dict["set_custom_workers"] and use_parallel:
        # Calculate optimal number of workers based on system resources
        optimal_workers = get_optimal_workers()

        # Get custom number from user
        max_workers_input = questionary.text(
            f"Enter number of workers (1-{multiprocessing.cpu_count()}):",
            default=str(optimal_workers),
            validate=lambda text: text.isdigit()
            and 1 <= int(text) <= multiprocessing.cpu_count(),
            style=custom_style,
        ).ask()
        max_workers = int(max_workers_input)
        print_info(f"Using {max_workers} parallel workers for processing")

    # Confirm settings
    print_info("\nYour selected settings:")
    print(f"Source directory: {source_dir}")
    print(f"Recursive mode: {recurse}")
    if recurse and use_parallel:
        print(f"Parallel processing: Yes ({max_workers} workers)")
    else:
        print("Parallel processing: No")
    print(f"Custom prefix: {prefix if prefix else 'Auto-detect'}")
    print(f"Generate full sample: {options_dict['gen_full']}")
    print(f"Time match: {options_dict['time_match']}")
    print(f"Generate chords: {options_dict['chords']}")

    if options_dict["chords"]:
        if chord_qualities:
            print(f"Chord qualities: {', '.join(chord_qualities)}")
        else:
            print("Generating all chord qualities")

        if selected_chord_types:
            print("Selected specific chord types:")
            for quality, types in selected_chord_types.items():
                print(f"  {quality}: {', '.join(types)}")
        else:
            print("Generating all chord types within selected qualities")

        print(f"Generate inversions: {generate_inversions}")
        if generate_inversions and selected_inversions is not None:
            if selected_inversions:
                print("Selected inversions:")
                # Group chords by their selected inversions for cleaner display
                inv_to_chords = {}
                if isinstance(selected_inversions, dict):  # Check if it's a dictionary
                    for (quality, chord_type), inv_list in selected_inversions.items():
                        inv_key = tuple(sorted(inv_list))  # Make the list hashable
                        if inv_key not in inv_to_chords:
                            inv_to_chords[inv_key] = []
                        inv_to_chords[inv_key].append(f"{quality} {chord_type}")

                    # Display grouped inversions
                    for inv_list, chords in inv_to_chords.items():
                        # Convert tuple to list before using list comprehension
                        inv_suffix_list = [
                            get_ordinal_suffix(i) for i in list(inv_list)
                        ]
                        inv_str = ", ".join(inv_suffix_list)
                        chord_str = ", ".join(chords)
                        print(f"  {inv_str} for: {chord_str}")
                else:
                    print(f"  Selected inversions: {selected_inversions}")
    print(f"Play notes: {options_dict['play']}")
    print(f"Overwrite existing: {options_dict['overwrite']}")
    print(f"Keep artifacts: {options_dict['keep_artifacts']}")

    if not questionary.confirm(
        "Proceed with these settings?", default=True, style=custom_style
    ).ask():
        print_info("Operation cancelled.")
        return

    # Process the directories
    if recurse:
        directories = [source_dir]

        # Walk through directories but skip "expansion", "exp_chords", and "exp" directories
        for root, dirs, files in os.walk(source_dir):
            # Skip traversing into output directories
            if "expansion" in dirs:
                dirs.remove("expansion")
            if "exp_chords" in dirs:
                dirs.remove("exp_chords")
            if "exp" in dirs:
                dirs.remove("exp")

            # Add only directories that don't have an "exp" subdirectory (unless overwrite is enabled)
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                exp_path = os.path.join(dir_path, "exp")

                # Skip directories that have already been processed (have an exp dir)
                # unless overwrite is enabled
                if os.path.exists(exp_path) and not options_dict["overwrite"]:
                    continue  # Skip this directory

                # Check if directory contains WAV files before adding it
                try:
                    wav_files_in_dir = [
                        f for f in os.listdir(dir_path) if f.lower().endswith(".wav")
                    ]

                    # Check if any of the WAV files contain valid note information
                    valid_samples = []
                    for sample in wav_files_in_dir:
                        note, octave = parse_note_from_filename(sample)
                        if note and octave:
                            valid_samples.append(sample)

                    if valid_samples:
                        print_info(
                            f"Found directory with {len(valid_samples)} valid note samples: {dir_path}"
                        )
                        directories.append(dir_path)
                    elif wav_files_in_dir:
                        print_info(
                            f"Skipping directory with WAV files but no valid note samples: {dir_path}"
                        )
                    else:
                        print_info(f"Skipping directory with no WAV files: {dir_path}")
                except (PermissionError, OSError) as e:
                    print_warning(f"Could not access directory {dir_path}: {str(e)}")
                    continue

        # Remove duplicates while preserving order
        seen = set()
        directories = [x for x in directories if not (x in seen or seen.add(x))]

        # Check if we have any directories to process
        if len(directories) <= 1:  # Only the source directory
            print_warning(
                "No subdirectories with valid note samples found for recursive processing."
            )
            print_info("Will only process the main source directory.")

        print(
            f"Found {len(directories)} directories to process (excluding expansion and output directories)"
        )

        # Process directories sequentially or in parallel
        if use_parallel and max_workers > 1:
            print_info(f"Processing directories in parallel with {max_workers} workers")

            # Create a list of processing tasks
            processing_tasks = []
            for directory in directories:
                # Create expansion subdirectory for output
                target_dir = os.path.join(directory, "expansion")

                # Delete existing expansion directory if overwrite is enabled
                if options_dict["overwrite"] and os.path.exists(target_dir):
                    print_info(f"Removing existing expansion directory: {target_dir}")
                    import shutil

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
                        "pitch_shift_method": pitch_shift_method,  # New pitch shift method parameter
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
                    pitch_shift_method=pitch_shift_method,  # Add pitch shift method parameter
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
            pitch_shift_method=pitch_shift_method,  # Add pitch shift method parameter
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
            overwrite=kwargs.get("overwrite", False),  # Add overwrite parameter
            pitch_shift_method=kwargs.get(
                "pitch_shift_method", "standard"
            ),  # Changed default to standard
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
