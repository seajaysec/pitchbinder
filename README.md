# Pitchbinder - Single notes to M8 sliced ranges and chords

A powerful tool for expanding a small set of samples into a complete 8-octave range, as well as any associated chords and inversions you'd like. Output is pre-sliced per root note, with full compatibility for the Dirtywave M8 and OcenAudio.

## Features

- **Intelligent Sample Generation**: Creates missing notes based on your existing samples.
- **Filename Detection**: Automatically detects note and octave from filenames.
- **Time Matching**: Optionally matches all generated samples to the average length of source samples.
- **Full Sample Generation**: Creates a single WAV file with all notes in sequence.
- **Slice Marking**: Embeds full sample with markers for each note, compatible with Dirtywave M8 and more.
- **Recursive Processing**: Process multiple directories at once.
- **Playback**: Option to play all generated notes in sequence after processing.
- **Interactive Interface**: User-friendly questionary-based interface for all options.
- **Granular Chord Selection**: Select specific chord qualities, chord types, and inversions with precision.
- **Optimized Menus**: Streamlined menu system for faster workflow and more intuitive options.

## How It Works

Pitchbinder analyzes your existing samples, identifies the notes they represent, and uses pitch-shifting algorithms to generate the missing notes. It finds the closest existing sample to use as a source for each new note, ensuring the most natural sound possible.

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

The Pitchbinder uses a questionary-based interactive interface that guides you through the process. Here's a step-by-step walkthrough:

1. **Source Directory Selection**
   ```
   Enter the source directory containing WAV samples: /path/to/your/samples/
   ```
   - Type the path to your sample directory
   - You can use tab-completion to navigate directories
   - The tool will scan and report how many WAV files it found

2. **Additional Options**
   ```
   Select additional options:
   > [ ] Generate a single WAV file with all notes in sequence
     [ ] Match all generated samples to the average length of source samples
     [ ] Generate chord samples
     [ ] Play all notes when done
     [ ] Overwrite existing expansion directories
     [ ] Keep all generated files (don't clean up artifacts)
     [ ] Use a custom prefix for generated files (Otherwise auto-detect)
     [ ] Set custom number of parallel workers
   ```
   - Use arrow keys to navigate
   - Press Space to select/deselect options
   - Press Enter when done

3. **Pitch Shifting Method**
   ```
   Which pitch shifting method would you like to use?
   > both (recommended for best quality)
     standard (faster, more consistent)
     granular (better for extreme shifts)
   ```
   - Select the most appropriate method for your use case

4. **Chord Generation Settings (if selected)**
   
   a. **Chord Qualities Selection**
   ```
   Which chord qualities would you like to generate?
   > [ ] Major
     [ ] Minor
     [ ] Diminished
     [ ] Augmented
     [ ] Suspended
     ...
   ```
   
   b. **Specific Chord Types Selection**
   ```
   Would you like to select specific chord types within these qualities? (y/n)
   ```
   
   c. **Chord Types Selection (per quality)**
   ```
   Which Major chord types would you like to generate?
   > [ ] Major fifth
     [ ] Major sixth
     [ ] Major seventh
     [ ] Major ninth
     ...
   ```
   
   d. **Inversions Selection**
   ```
   Would you also like to generate chord inversions? (y/n)
   ```
   
   e. **Granular Inversion Selection**
   ```
   Generate all possible inversions for each chord? (y/n)
   ```
   
   f. **Per-Chord Inversion Selection (if not generating all)**
   ```
   Select inversions for Major seventh (Major) - 4-note chord:
   > [ ] 1st
     [ ] 2nd
     [ ] 3rd
   ```

5. **Review and Confirm Settings**

   The tool will display a summary of your selected settings before proceeding.

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

## Changelog

### v1.1
- Renamed the project to "Pitchbinder" to better reflect its purpose
- Completely streamlined the menu system for a more intuitive workflow
- Added granular chord selection allowing specific chord qualities, types, and inversions
- Improved handling of recursive processing
- Enhanced the pitch shifting algorithm options

### v1.0
- Initial release with basic note expansion functionality
- Simple chord generation
- Basic recursive processing capabilities