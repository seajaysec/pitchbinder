# Note Expander

A powerful tool for expanding a small set of piano samples into a complete 8-octave range.

## Overview

Note Expander takes a few piano samples with note information in their filenames and intelligently generates the missing notes across the full piano range (C1-B8). It uses high-quality pitch shifting and optional time stretching to create natural-sounding samples for all 88 piano keys.

## Why Expand Notes

Traditional audio tools and DAWs offer pitch-shifting capabilities, but they come with significant limitations when you're trying to create a complete instrument:

- **Preserving Natural Duration**: Standard pitch-shifting typically speeds up samples when pitching higher and slows them down when pitching lower. This creates unnatural results when attempting to build an instrument across multiple octaves, as high notes become too short and low notes too long.

- **Limited Range**: Many pitch-shifting algorithms struggle with extreme shifts (e.g., moving multiple octaves from the source). The audio quality deteriorates quickly, introducing artifacts and losing the original character of the instrument.

- **Clunky Chord Creation**: Creating chords often requires manually placing and tuning multiple samples. This process is time-consuming and difficult to adjust, especially when working with multiple chord types and inversions.

- **Workflow Complexity**: Setting up a full instrument with proper velocity layers, round-robin samples, and chord variations can require hundreds of manual adjustments in traditional workflows.

Note Expander addresses these issues by:

1. **Smart Sample Selection**: For each generated note, it finds the closest source sample to minimize pitch-shifting artifacts.

2. **Advanced Time Stretching**: Using Librosa's high-quality time stretching algorithms, it preserves the natural duration of notes regardless of pitch, resulting in a more consistent instrument feel.

3. **Intelligent Envelope Handling**: The tool applies gentle envelopes to ensure smooth attack and decay characteristics are maintained across all generated samples.

4. **Automated Chord Generation**: Rather than manually constructing chords, Note Expander can automatically generate common chord types and inversions from your source samples.

5. **FFT Size Optimization**: For very short samples, the tool automatically adjusts the FFT window size to prevent spectral artifacts that would otherwise occur during processing.

While this approach sacrifices a small amount of audio fidelity compared to using recordings of every single note, it offers a pragmatic balance between quality and practicality. The technical choices made in the implementation—such as using spectral-based time stretching, adaptive FFT windows, and envelope preservation—ensure that the results remain musically useful even for critical applications.

## Features

- **Intelligent Sample Generation**: Creates missing notes based on your existing samples.
- **Filename Detection**: Automatically detects note and octave from filenames.
- **Time Matching**: Optionally matches all generated samples to the average length of source samples.
- **Full Sample Generation**: Creates a single WAV file with all notes in sequence.
- **Slice Marking**: Embeds full sample with markers for each note, compatible with Dirtywave M8 and more.
- **Recursive Processing**: Process multiple directories at once.
- **Playback**: Option to play all generated notes in sequence after processing.
- **Interactive Interface**: User-friendly questionary-based interface for all options.

## How It Works

Note Expander analyzes your existing samples, identifies the notes they represent, and uses pitch-shifting algorithms to generate the missing notes. It finds the closest existing sample to use as a source for each new note, ensuring the most natural sound possible.

### Supported Filename Formats

The script can detect notes and octaves from filenames in formats like:
- `Piano-C4.wav`
- `Meowsic-F#3.wav`
- `CP70-Ab2.wav`

As long as the filename contains a note letter (A-G), optional sharp (#), optional flat (b), and octave number, it will be detected.

## Installation

1. Clone this repository or download the script
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Simply run the script and follow the interactive prompts:
```
python note_expander.py
```

The interactive interface will guide you through all available options:
- Selecting source directory
- Processing subdirectories recursively
- Setting a custom prefix for generated files
- Generating a full sample file with all notes
- Time matching samples
- Generating chord samples
- Playing all notes when done
- Overwriting existing expansion directories

## Interactive Interface Guide

The note-expander uses a questionary-based interactive interface that guides you through the process. Here's a step-by-step walkthrough:

1. **Source Directory Selection**
   ```
   Enter the source directory containing WAV samples: /path/to/your/samples/
   ```
   - Type the path to your sample directory
   - You can use tab-completion to navigate directories
   - The tool will scan and report how many WAV files it found

2. **Recursive Processing**
   ```
   Process all subdirectories recursively? (y/n)
   ```
   - Choose 'y' to process all subdirectories containing samples
   - Choose 'n' to only process the main directory

3. **Custom Prefix**
   ```
   Use a custom prefix for generated files? (Otherwise auto-detect) (y/n)
   ```
   - If you select 'y', you'll be prompted to enter a custom prefix
   - If you select 'n', the prefix will be automatically detected from your existing samples

4. **Additional Options**
   ```
   Select additional options:
   > [ ] Generate a single WAV file with all notes in sequence
     [ ] Match all generated samples to the average length of source samples
     [ ] Generate chord samples
     [ ] Play all notes when done
     [ ] Overwrite existing expansion directories
     [ ] Keep all generated files (don't clean up artifacts)
   ```
   - Use arrow keys to navigate
   - Press Space to select/deselect options
   - Press Enter when done
   
5. **Chord Generation (if selected)**
   
   If you selected "Generate chord samples", you'll see additional prompts:
   
   a. **Chord Generation Mode**
   ```
   How would you like to generate chords?
   > Generate all chord types
     Select specific chord qualities
   ```
   
   b. **Chord Qualities (if "Select specific chord qualities" is chosen)**
   ```
   Select chord qualities to generate:
   > [ ] Major
     [ ] Minor
     [ ] Diminished
     [ ] Augmented
     [ ] Suspended
     ...
   ```
   
   c. **Inversions**
   ```
   Generate chord inversions? (This will create all possible inversions for each chord) (y/n)
   ```

6. **Review and Confirm Settings**

   The tool will display a summary of your selected settings:
   ```
   Your selected settings:
   Source directory: /path/to/your/samples/
   Recursive mode: False
   Custom prefix: Auto-detect
   Generate full sample: True
   Time match: True
   Generate chords: True
   Chord qualities: Major, Minor
   Generate inversions: True
   Play notes: False
   Overwrite existing: False
   Keep artifacts: False
   
   Proceed with these settings? (y/n)
   ```
   
   - Review the settings and confirm to proceed
   - If you select 'n', the process will be canceled

7. **Processing**

   Once confirmed, the tool will begin processing your samples:
   - It will create an "expansion" directory within your source directory
   - All generated files will be placed there, organized by type
   - Progress bars will show the status of each step
   - When completed, you'll see a success message

## Output

All generated samples are saved to an `expansion` subdirectory within each processed directory. This keeps your original samples untouched while organizing the new ones.

When generating a full sample file, a single WAV file containing all notes in sequence is created, along with a CUE file that marks the position of each note.

## Tips for Best Results

1. **Use High-Quality Source Samples**: The better your source samples, the better the generated ones will be.
2. **Provide Samples Across the Range**: For best results, provide samples that span the range (e.g., one sample per octave).
3. **Consistent Naming**: Make sure your sample filenames follow a consistent pattern with clear note information.
4. **Time Matching**: Use the time matching option if you want all samples to have the same duration.

## Limitations

- The quality of pitch-shifted samples decreases as they get further from the source sample.
- Very extreme pitch shifts (e.g., shifting a C1 to a C8) may not sound natural.

## Troubleshooting

### "Directory not empty" Error During Cleanup

If you encounter an error like this during the cleanup phase:

```
OSError: [Errno 66] Directory not empty: '/path/to/your/samples/exp_chords'
```

This can happen when the script tries to remove temporary directories but encounters locked files or permission issues. You can:

1. **Keep artifacts**: Select the "Keep all generated files" option when running the script to skip cleanup entirely.

2. **Manual cleanup**: The error doesn't affect the generated samples, so you can manually delete the `exp_chords` directory later.

3. **Check permissions**: Ensure you have full read/write access to the directory and its contents.

4. **Close other applications**: Make sure no other applications (DAWs, audio players, etc.) are accessing files in the directory.

5. **Restart and try again**: Sometimes simply running the script again will succeed if the issue was temporary.

### Unable to Detect Notes in Filenames

If the script reports that it cannot detect notes in your sample filenames:

1. **Check format**: Ensure filenames contain a note letter (A-G), followed by an optional sharp (#) or flat (b), and then an octave number (e.g., `Piano-C4.wav`, `Synth-Ab3.wav`).

2. **Rename files**: You may need to rename your samples to follow the supported naming conventions.

3. **Remove special characters**: Some special characters in filenames can interfere with note detection.

### Other Issues

For other problems:

- Try processing a smaller set of samples to identify if specific files are causing issues
- Check the terminal output for warning messages that might provide more information
- Ensure your Python environment has all the required dependencies installed
- Try running with the `-v` or `--verbose` flag for more detailed output