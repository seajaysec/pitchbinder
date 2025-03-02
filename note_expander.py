import argparse
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
    return [f for f in os.listdir(directory) if f.lower().endswith(".wav") and "-00-Full" not in f]


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
                        n_fft = 2**int(np.log2(len(new_audio) - 1))
                        n_fft = max(32, n_fft)  # Ensure it's not too small
                        print(f"  Using smaller FFT window (n_fft={n_fft}) for short sample")
                        
                    # Use librosa's high-quality time stretching with corrected rate and appropriate n_fft
                    new_audio = librosa.effects.time_stretch(
                        new_audio, rate=rate, n_fft=n_fft
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
                hop_length=128
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
        with wave.open(output_path, 'rb') as wav_read:
            params = wav_read.getparams()
            frames = wav_read.readframes(wav_read.getnframes())
            
        # Create a new WAV file with cue markers
        with wave.open(output_path + '.temp', 'wb') as wav_write:
            wav_write.setparams(params)
            
            # Write the audio data
            wav_write.writeframes(frames)
            
            # Add cue chunk
            cue_chunk = create_cue_chunk(cue_positions)
            
            # We need to manually add the cue chunk to the file
            # This is a bit hacky but necessary since wave module doesn't support cue chunks
            with open(output_path + '.temp', 'ab') as f:
                f.write(b'cue ')  # Chunk ID
                f.write(struct.pack('<I', len(cue_chunk)))  # Chunk size
                f.write(cue_chunk)  # Chunk data
        
        # Replace the original file with the new one
        os.replace(output_path + '.temp', output_path)
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
    chunk_data = struct.pack('<I', num_cues)
    
    # Add each cue point
    for i, (label, position) in enumerate(cue_positions):
        cue_id = i + 1
        position_samples = position
        
        # Cue point structure:
        # ID (4 bytes) + Position (4 bytes) + Data Chunk ID (4 bytes) +
        # Chunk Start (4 bytes) + Block Start (4 bytes) + Sample Offset (4 bytes)
        cue_point = struct.pack('<II4sIII', 
                               cue_id,                # ID
                               position_samples,      # Position
                               b'data',               # Data Chunk ID
                               0,                     # Chunk Start
                               0,                     # Block Start
                               position_samples)      # Sample Offset
        
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
    source_dir, target_dir, prefix=None, play=False, gen_full=False, time_match=False
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
    args = parser.parse_args()

    # If recursive mode is enabled, find all directories to process
    if args.recurse:
        directories = [args.source]
        
        # Walk through directories but skip "expansion" directories
        for root, dirs, files in os.walk(args.source):
            # Remove 'expansion' from dirs to prevent os.walk from traversing into it
            if 'expansion' in dirs:
                dirs.remove('expansion')
                
            for dir_name in dirs:
                directories.append(os.path.join(root, dir_name))

        print(f"Found {len(directories)} directories to process (excluding expansion directories)")
        
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
        )


if __name__ == "__main__":
    main()
