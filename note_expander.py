import argparse
import csv
import os
import re

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal


def parse_note_from_filename(filename):
    """Extract note and octave from filename."""
    # Match patterns like 'CC Piano-G#5.wav' or 'CC Piano-G3.wav'
    match = re.search(r"([A-G]#?)(\d+)", filename)
    if match:
        note, octave = match.groups()
        return note, int(octave)
    return None, None


def get_note_frequency(note, octave):
    """Calculate frequency for a given note and octave."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_index = notes.index(note)

    # A4 is 440Hz
    a4_freq = 440.0
    a4_index = notes.index("A") + (4 * 12)
    note_index_full = note_index + (octave * 12)

    # Calculate frequency using equal temperament formula
    half_steps = note_index_full - a4_index
    return a4_freq * (2 ** (half_steps / 12.0))


def get_all_wav_files(directory="."):
    """Get all WAV files in the specified directory."""
    # Exclude files with -00-Full in the filename
    return [
        f
        for f in os.listdir(directory)
        if f.lower().endswith(".wav") and "-00-Full" not in f
    ]


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
        return signal.resample(audio_data, new_length), sr

    return audio_data, sr


def find_closest_sample(target_note, target_octave, existing_samples):
    """Find the closest existing sample to use as a source."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    target_index = notes.index(target_note) + (target_octave * 12)

    closest_sample = None
    min_distance = float("inf")

    for sample in existing_samples:
        note, octave = parse_note_from_filename(sample)
        if note and octave:
            sample_index = notes.index(note) + (octave * 12)
            distance = abs(target_index - sample_index)

            if distance < min_distance:
                min_distance = distance
                closest_sample = sample

    return closest_sample


# Add chord definitions from Chords.tsv
CHORD_DEFINITIONS = [
    # Format: (Chord name, Quality, Semitones list, Notes count)
    ("Major chord", "Major", [0, 4, 7], 3),
    ("Dominant seventh chord", "Major", [0, 4, 7, 10], 4),
    ("Major seventh chord", "Major", [0, 4, 7, 11], 4),
    ("Major sixth chord", "Major", [0, 4, 7, 9], 4),
    ("Dominant Minor ninth", "Major", [0, 4, 7, 10, 13], 5),
    ("Dominant ninth", "Major", [0, 4, 7, 10, 14], 5),
    ("Dominant seventh sharp nine (Hendrix)", "Major", [0, 4, 7, 10, 15], 5),
    ("Lydian chord", "Major", [0, 4, 7, 11, 18], 5),
    ("Major sixth ninth chord", "Major", [0, 4, 7, 9, 14], 5),
    ("Major ninth chord", "Major", [0, 4, 7, 11, 14], 5),
    ("Seven six chord", "Major", [0, 4, 7, 9, 10], 5),
    ("Augmented eleventh chord", "Major", [0, 4, 7, 10, 14, 18], 6),
    ("Dominant eleventh chord", "Major", [0, 4, 7, 10, 14, 17], 6),
    ("Major eleventh chord", "Major", [0, 4, 7, 11, 14, 17], 6),
    ("Thirteenth flat ninth chord", "Major", [0, 4, 7, 10, 13, 21], 6),
    ("Dominant thirteenth chord", "Major", [0, 4, 7, 10, 14, 17, 21], 7),
    ("Major thirteenth chord", "Major", [0, 4, 7, 11, 14, 17, 21], 7),
    ("Minor chord", "Minor", [0, 3, 7], 3),
    ("Minor Major seventh chord", "Minor", [0, 3, 7, 11], 4),
    ("Minor seventh chord", "Minor", [0, 3, 7, 10], 4),
    ("Minor sixth chord", "Minor", [0, 3, 7, 9], 4),
    ("Minor ninth chord", "Minor", [0, 3, 7, 10, 14], 5),
    ("Minor sixth ninth chord", "Minor", [0, 3, 7, 9, 14], 5),
    ("Minor eleventh chord", "Minor", [0, 3, 7, 10, 14, 17], 6),
    ("Minor thirteenth chord", "Minor", [0, 3, 7, 10, 14, 17, 21], 7),
    ("Augmented chord", "Augmented", [0, 4, 8], 3),
    ("Augmented Major seventh chord", "Augmented", [0, 4, 8, 11], 4),
    ("Augmented seventh chord", "Augmented", [0, 4, 8, 10], 4),
    ("Major seventh sharp eleventh chord", "Augmented", [0, 4, 8, 11, 18], 5),
    ("Ninth Augmented fifth chord", "Augmented", [0, 4, 8, 10, 14], 5),
    ("Diminished chord", "Diminished", [0, 3, 6], 3),
    ("Diminished Major seventh chord", "Diminished", [0, 3, 6, 11], 4),
    ("Diminished seventh chord", "Diminished", [0, 3, 6, 9], 4),
    ("Half-Diminished seventh chord", "Diminished", [0, 4, 6, 10], 4),
    ("Power chord", "Indeterminate", [0, 7], 2),
    ("Augmented sixth chord (Italian)", "Predominant", [0, 4, 10], 3),
    ("Augmented sixth chord (French)", "Predominant", [0, 4, 6, 10], 4),
    ("Augmented sixth chord (German)", "Predominant", [0, 4, 7, 10], 4),
    ("Tristan chord", "Predominant", [0, 3, 6, 10], 4),
    ("Suspended chord", "Suspended", [0, 5, 7], 3),
    ("Seventh suspension four chord", "Suspended", [0, 5, 7, 10], 4),
    ("Ninth flat fifth chord", "M3+d5", [0, 4, 6, 10, 14], 5),
    ("Thirteenth flat ninth flat fifth chord", "M3+d5", [0, 4, 6, 10, 13, 21], 6),
    ("Dream chord", "Just", [0, 5, 6, 7], 4),
    ("Magic chord", "Just", [0, 1, 5, 6, 10, 12, 15, 17], 8),
    ("Elektra chord", "Bitonal", [0, 7, 9, 13, 16], 5),
    ("So What chord", "Bitonal", [0, 5, 10, 15, 19], 5),
    ("Petrushka chord", "Bitonal", [0, 1, 4, 6, 7, 10], 6),
    ("Farben chord", "Atonal", [0, 8, 11, 16, 21], 5),
    ("Viennese trichord", "Atonal", [0, 1, 6], 3),
    ("Viennese trichord (alt)", "Atonal", [0, 6, 7], 3),
    ("Mystic chord", "Atonal", [0, 6, 10, 16, 21, 26], 6),
    ("Ode-to-Napoleon hexachord", "Atonal", [0, 1, 4, 5, 8, 9], 6),
    ("Northern lights chord", "Atonal", [1, 2, 8, 12, 15, 18, 19, 22, 23, 28, 31], 11),
]


def load_chord_definitions_from_tsv(tsv_path):
    """Load chord definitions from a TSV file."""
    chord_defs = []
    try:
        with open(tsv_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 4:
                    chord_names = row[0].strip('"').split(", ")
                    quality = row[1]
                    # Parse semitones from string like "0, 4, 7"
                    semitones_str = row[2].strip('"')
                    semitones = [int(s.strip()) for s in semitones_str.split(",")]
                    notes_count = int(row[3])

                    # Add each chord name as a separate entry
                    for chord_name in chord_names:
                        chord_defs.append((chord_name, quality, semitones, notes_count))
        return chord_defs
    except Exception as e:
        print(f"Error loading chord definitions from {tsv_path}: {e}")
        print("Using built-in chord definitions instead.")
        return CHORD_DEFINITIONS


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
        print(
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

        # Check if the file exists in target directory first, then source
        note_path = None
        if os.path.exists(os.path.join(target_dir, note_filename)):
            note_path = os.path.join(target_dir, note_filename)
        elif os.path.exists(os.path.join(source_dir, note_filename)):
            note_path = os.path.join(source_dir, note_filename)
        else:
            # If the exact note doesn't exist, find the closest available note
            closest_sample = find_closest_sample(note, octave, all_samples)
            if closest_sample:
                closest_note, closest_octave = parse_note_from_filename(closest_sample)

                # Load the closest sample
                audio, sr = librosa.load(
                    os.path.join(source_dir, closest_sample), sr=None
                )

                # Pitch shift to the target note
                audio, sr = pitch_shift_sample(
                    audio, sr, closest_note, closest_octave, note, octave
                )

                note_audios.append(audio)
                max_length = max(max_length, len(audio))
                continue
            else:
                print(
                    f"Warning: Could not find a suitable sample for {note}{octave} in chord"
                )
                continue

        # Load the audio for this note
        audio, _ = librosa.load(note_path, sr=sr)
        note_audios.append(audio)
        max_length = max(max_length, len(audio))

    if not note_audios:
        print(
            f"Error: Could not generate chord with root {root_note}{root_octave} - no valid notes found"
        )
        return None, None

    # Extend chord duration by the specified factor
    if chord_duration_factor > 1.0:
        extended_max_length = int(max_length * chord_duration_factor)
        extended_note_audios = []

        for audio in note_audios:
            # Use time stretching to extend the duration
            rate = 1.0 / chord_duration_factor  # Inverse for librosa's time_stretch

            # For very short samples, use a smaller n_fft value
            n_fft = 2048  # Default value
            if len(audio) < n_fft:
                # Use a power of 2 that's smaller than the audio length
                n_fft = 2 ** int(np.log2(len(audio) - 1))
                n_fft = max(32, n_fft)  # Ensure it's not too small

            # Convert audio to float64 to ensure it's the right type for time_stretch
            audio_float = audio.astype(np.float64)

            # Use librosa's time stretching
            extended_audio = librosa.effects.time_stretch(
                audio_float, rate=float(rate), n_fft=n_fft
            )

            extended_note_audios.append(extended_audio)

        note_audios = extended_note_audios
        max_length = extended_max_length

    # Mix all notes together
    mixed_audio = np.zeros(max_length)
    for audio in note_audios:
        # Pad shorter audio to match the longest one
        padded_audio = np.pad(audio, (0, max(0, max_length - len(audio))))

        # Add to the mix (with normalization to prevent clipping)
        mixed_audio += padded_audio / len(note_audios)

    # Apply a slight fade-in and fade-out to prevent clicks
    fade_samples = min(int(sr * 0.01), max_length // 10)  # 10ms fade or 1/10 of length
    if fade_samples > 0:
        # Fade in
        fade_in = np.linspace(0, 1, fade_samples)
        mixed_audio[:fade_samples] *= fade_in

        # Fade out
        fade_out = np.linspace(1, 0, fade_samples)
        mixed_audio[-fade_samples:] *= fade_out

    return mixed_audio, sr


def generate_chords(
    prefix,
    all_samples,
    source_dir,
    chord_dir,
    target_dir,
    chord_definitions=None,
    tsv_path=None,
):
    """Generate chord samples based on the provided chord definitions."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Load chord definitions from TSV if provided
    if tsv_path and os.path.exists(tsv_path):
        chord_defs = load_chord_definitions_from_tsv(tsv_path)
    else:
        chord_defs = chord_definitions or CHORD_DEFINITIONS

    print(f"\nGenerating {len(chord_defs)} chord types...")

    # Create the main chord directory if it doesn't exist
    if not os.path.exists(chord_dir):
        os.makedirs(chord_dir)

    # Group chord definitions by quality
    chord_by_quality = {}
    for chord_name, quality, semitones, _ in chord_defs:
        if quality not in chord_by_quality:
            chord_by_quality[quality] = []
        chord_by_quality[quality].append((chord_name, semitones))

    # Generate chords for each quality
    for quality, chords in chord_by_quality.items():
        # Create a directory for this chord quality
        quality_dir = os.path.join(chord_dir, quality)
        if not os.path.exists(quality_dir):
            os.makedirs(quality_dir)

        print(f"\nGenerating {len(chords)} {quality} chord types...")

        # Generate each chord type with roots from C2 to B4
        for chord_name, semitones in chords:
            # Create a safe filename from the chord name
            safe_chord_name = re.sub(r"[^\w\-]", "_", chord_name)

            print(f"  Generating {chord_name} chords...")

            # Generate chords with roots from C2 to B4
            for octave in range(2, 5):
                for note in notes:
                    # Skip if the highest note in the chord would be above B8
                    highest_semitone = max(semitones)
                    highest_note, highest_octave = get_note_from_semitone(
                        note, octave, highest_semitone
                    )
                    if highest_octave > 8:
                        continue

                    # Generate the chord - pass both source and target dirs
                    chord_audio, sr = generate_chord(
                        note,
                        octave,
                        semitones,
                        all_samples,
                        source_dir,
                        target_dir,  # Pass the expansion directory
                        prefix,
                        chord_duration_factor=4.0,
                    )

                    if chord_audio is not None:
                        # Save the chord
                        chord_filename = (
                            f"{prefix}-{safe_chord_name}-{note}{octave}.wav"
                        )
                        chord_path = os.path.join(quality_dir, chord_filename)
                        sf.write(chord_path, chord_audio, sr)
                        print(f"    Generated {chord_filename}")

    print("\nChord generation complete!")
    return chord_dir


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

    # Calculate average duration of existing samples more accurately
    durations = []
    for sample in existing_samples:
        if sample in parsed_samples:
            # Get file info directly using soundfile instead of loading the entire file
            file_path = os.path.join(source_dir, sample)
            info = sf.info(file_path)
            duration = info.duration  # This gives the exact duration in seconds
            durations.append(duration)
            print(f"Sample {sample} duration: {duration:.2f} seconds")

    avg_duration = np.mean(durations) if durations else 2.0  # Default to 2 seconds

    if time_match and durations:
        print(f"Time matching enabled - target duration: {avg_duration:.2f} seconds")

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
                print(f"No suitable source sample found for {target_filename}")
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
                print(
                    f"  Before stretching: {target_filename} duration: {current_duration:.2f} seconds"
                )

                # In librosa's time_stretch, rate > 1 speeds up, rate < 1 slows down
                # So we need to use 1/stretch_factor to get the correct behavior
                stretch_factor = avg_duration / current_duration
                rate = 1.0 / stretch_factor  # Inverse for librosa's time_stretch

                if (
                    abs(stretch_factor - 1.0) > 0.01
                ):  # Only stretch if difference is significant
                    print(
                        f"  Time stretching {target_filename} (target: {avg_duration:.2f}s, factor: {stretch_factor:.2f}, rate: {rate:.2f})"
                    )

                    # For very short samples, use a smaller n_fft value
                    n_fft = 2048  # Default value
                    if len(new_audio) < n_fft:
                        # Use a power of 2 that's smaller than the audio length
                        n_fft = 2 ** int(np.log2(len(new_audio) - 1))
                        n_fft = max(32, n_fft)  # Ensure it's not too small
                        print(
                            f"  Using smaller FFT window (n_fft={n_fft}) for short sample"
                        )

                    # Use librosa's high-quality time stretching with corrected rate and appropriate n_fft
                    # Convert to float64 to ensure correct type for time_stretch
                    new_audio_float = new_audio.astype(np.float64)
                    new_audio = librosa.effects.time_stretch(
                        new_audio_float, rate=float(rate), n_fft=n_fft
                    )

                    # Verify the new duration
                    new_duration = len(new_audio) / new_sr
                    print(
                        f"  After stretching: {target_filename} duration: {new_duration:.2f} seconds"
                    )

            # Save new sample to target directory
            sf.write(os.path.join(target_dir, target_filename), new_audio, new_sr)
            print(f"Generated {target_filename} from {source_sample}")
            generated_files.append(target_filename)

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
    """Generate a single WAV file with all notes in sequence and embedded cue markers."""
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

    # Track positions for cue markers
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
            non_silent = librosa.effects.trim(
                audio,
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

    # Save the combined audio to the source directory (not the expansion directory)
    output_filename = f"{prefix}-00-Full.wav"
    output_path = os.path.join(source_dir, output_filename)

    # First save the audio data
    sf.write(output_path, combined_audio, sr)
    print(f"Generated full sample file: {output_filename} (in source directory)")

    # Now add cue markers to the WAV file
    try:
        # Open the WAV file for reading and writing in binary mode
        with wave.open(output_path, "rb") as wav_read:
            params = wav_read.getparams()
            frames = wav_read.readframes(wav_read.getnframes())

        # Create a new WAV file with cue markers
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
        print(f"Added {len(cue_positions)} cue markers to {output_filename}")

    except Exception as e:
        print(f"Warning: Could not add cue markers to WAV file: {e}")

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


def generate_cue_file(cue_positions, prefix, wav_filename, target_dir, sample_rate):
    """Generate a separate CUE file as a fallback."""
    cue_filename = f"{prefix}-00-Full.cue"
    cue_path = os.path.join(target_dir, cue_filename)

    with open(cue_path, "w") as cue_file:
        cue_file.write(f'TITLE "{prefix} Full Sample"\n')
        cue_file.write(f'FILE "{wav_filename}" WAVE\n')

        for i, (note_name, position) in enumerate(cue_positions, 1):
            # Convert sample position to time
            position_seconds = position / sample_rate
            minutes, seconds = divmod(position_seconds, 60)
            frames = int((seconds % 1) * 75)  # CUE uses 75 frames per second
            seconds = int(seconds)

            cue_file.write(f"  TRACK {i:02d} AUDIO\n")
            cue_file.write(f'    TITLE "Note {note_name}"\n')
            cue_file.write(
                f"    INDEX 01 {int(minutes):02d}:{seconds:02d}:{frames:02d}\n"
            )

    print(f"Generated CUE file: {cue_filename}")


def process_directory(
    source_dir,
    target_dir,
    prefix=None,
    play=False,
    gen_full=False,
    time_match=False,
    gen_chords=False,
    chord_tsv=None,
):
    """Process a single directory to generate missing samples."""
    print(f"\n{'='*60}")
    print(f"Processing directory: {source_dir}")
    print(f"{'='*60}")

    # Get all WAV files in the source directory
    existing_samples = get_all_wav_files(source_dir)

    if not existing_samples:
        print(f"No WAV files found in the source directory: {source_dir}")
        return

    # Check if files have detectable notes
    valid_samples = []
    for sample in existing_samples:
        note, octave = parse_note_from_filename(sample)
        if note and octave:
            valid_samples.append(sample)

    if not valid_samples:
        print(f"No samples with detectable notes found in: {source_dir}")
        return

    if len(valid_samples) < len(existing_samples):
        print(
            f"Warning: {len(existing_samples) - len(valid_samples)} samples have undetectable notes and will be ignored"
        )

    # Auto-detect prefix if not provided
    dir_prefix = prefix
    if not dir_prefix:
        # Use the prefix from the first valid sample
        match = re.match(r"(.+)-[A-G]#?\d+\.wav", valid_samples[0])
        if match:
            dir_prefix = match.group(1)
        else:
            # Use the directory name as a fallback
            dir_prefix = os.path.basename(source_dir)
            if not dir_prefix:  # In case it's the root directory
                dir_prefix = "Piano"

    # Now that we've validated the samples, create the target directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"Using prefix: {dir_prefix}")
    print(f"Found {len(valid_samples)} valid samples in {source_dir}")
    print(f"Generated samples will be saved to {target_dir}")

    # Generate missing samples
    all_samples = generate_missing_samples(
        dir_prefix, valid_samples, source_dir, target_dir, time_match
    )

    print(f"\nGeneration complete. {len(all_samples)} total samples available.")

    # Generate chord samples if requested
    if gen_chords:
        chord_dir = os.path.join(source_dir, "exp_chords")
        generate_chords(
            dir_prefix,
            all_samples,
            source_dir,
            chord_dir,
            target_dir,  # Pass the expansion directory
            tsv_path=chord_tsv,
        )

    # Generate full sample file if requested
    full_sample_filename = None
    if gen_full:
        full_sample_filename = generate_full_sample(
            all_samples, dir_prefix, source_dir, target_dir
        )

    # Play all notes if requested (excluding the full sample)
    if play:
        # Filter out the full sample if it was generated
        samples_to_play = [s for s in all_samples if s != full_sample_filename]
        play_all_notes(samples_to_play, source_dir, target_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Expand piano samples to cover full octave range",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show defaults in help
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=".",
        help="Source directory containing WAV samples (default: current directory)",
    )
    parser.add_argument(
        "-r",
        "--recurse",
        action="store_true",
        help="Process all subdirectories recursively",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        default=None,
        help="Prefix for generated files (default: auto-detect)",
    )
    parser.add_argument(
        "-P",
        "--play",
        action="store_true",
        help="Play all notes when done",
    )
    parser.add_argument(
        "-f",
        "--gen-full",
        action="store_true",
        help="Generate a single WAV file with all notes in sequence",
    )
    parser.add_argument(
        "-t",
        "--time-match",
        action="store_true",
        help="Match all generated samples to the average length of source samples",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing expansion directories and regenerate all samples",
    )
    parser.add_argument(
        "-c",
        "--gen-chords",
        action="store_true",
        help="Generate chord samples based on chord definitions",
    )
    parser.add_argument(
        "--chord-tsv",
        default=None,
        help="Path to TSV file containing chord definitions (optional)",
    )
    args = parser.parse_args()

    # If recursive mode is enabled, find all directories to process
    if args.recurse:
        directories = [args.source]

        # Walk through directories but skip "expansion" directories
        for root, dirs, files in os.walk(args.source):
            # Remove 'expansion' from dirs to prevent os.walk from traversing into it
            if "expansion" in dirs:
                dirs.remove("expansion")

            for dir_name in dirs:
                directories.append(os.path.join(root, dir_name))

        print(
            f"Found {len(directories)} directories to process (excluding expansion directories)"
        )

        # Process each directory
        for directory in directories:
            # Create expansion subdirectory for output
            target_dir = os.path.join(directory, "expansion")

            # Delete existing expansion directory if overwrite is enabled
            if args.overwrite and os.path.exists(target_dir):
                print(f"Removing existing expansion directory: {target_dir}")
                import shutil

                shutil.rmtree(target_dir)

            process_directory(
                source_dir=directory,
                target_dir=target_dir,
                prefix=args.prefix,
                play=args.play,
                gen_full=args.gen_full,
                time_match=args.time_match,
                gen_chords=args.gen_chords,
                chord_tsv=args.chord_tsv,
            )
    else:
        # Process just the single directory
        # Create expansion subdirectory for output
        target_dir = os.path.join(args.source, "expansion")

        # Delete existing expansion directory if overwrite is enabled
        if args.overwrite and os.path.exists(target_dir):
            print(f"Removing existing expansion directory: {target_dir}")
            import shutil

            shutil.rmtree(target_dir)

        process_directory(
            source_dir=args.source,
            target_dir=target_dir,
            prefix=args.prefix,
            play=args.play,
            gen_full=args.gen_full,
            time_match=args.time_match,
            gen_chords=args.gen_chords,
            chord_tsv=args.chord_tsv,
        )


if __name__ == "__main__":
    main()
