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
- **Enhanced Chord Generation**: Fine-grained control over chord types and inversions (new in v1.1).

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
- Generating chord samples with fine-grained control
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

2. **Recursive Processing**
   Recursive processing is enabled by default to process all subdirectories.

3. **Custom Prefix**
   ```
   Use a custom prefix for generated files? (Otherwise auto-detect) (y/n)
   ```
   - If you select this option, you'll be prompted to enter a custom prefix
   - If not selected, the prefix will be automatically detected from your existing samples

4. **Additional Options**
   ```
   Select additional options:
   > [x] Generate a single WAV file with all notes in sequence
     [x] Match all generated samples to the average length of source samples
     [x] Generate chord samples
     [ ] Play all notes when done
     [ ] Overwrite existing expansion directories
     [ ] Keep all generated files (don't clean up artifacts)
     [ ] Use a custom prefix for generated files (Otherwise auto-detect)
     [ ] Set custom number of parallel workers
   ```
   - Use arrow keys to navigate
   - Press Space to select/deselect options
   - Press Enter when done
   
5. **Pitch Shifting Method**
   ```
   Which pitch shifting method would you like to use?
   > both (recommended for best quality)
     standard (faster, more consistent)
     granular (better for extreme shifts)
   ```
   
6. **Enhanced Chord Generation (new in v1.1)**
   
   If you selected "Generate chord samples", you'll see the improved interface:
   
   a. **Chord Qualities Selection**
   ```
   Which chord qualities would you like to generate?
   > [x] Major
     [x] Minor
     [x] Diminished
     [x] Augmented
     [x] Suspended
     ...
   ```
   
   b. **Specific Chord Types Selection**
   ```
   Would you like to select specific chord types within these qualities? (y/n)
   ```
   
   If yes:
   ```
   Which Major chord types would you like to generate?
   > [x] Major fifth
     [x] Major sixth
     [x] Major seventh
     ...
   ```
   
   c. **Inversions Control**
   ```
   Would you also like to generate chord inversions? (y/n)
   ```
   
   If yes:
   ```
   Generate all possible inversions for each chord? (y/n)
   ```
   
   If not generating all inversions:
   ```
   Select inversions for Major seventh (Major) - 4-note chord:
   > [x] 1st
     [x] 2nd
     [x] 3rd
   ```
   
   This granular selection allows you to precisely control which chord types and inversions are generated.

7. **Review and Confirm Settings**

   The tool will display a summary of your selected settings:
   ```
   Your selected settings:
   Source directory: /path/to/your/samples/
   Recursive mode: True
   Parallel processing: Yes (8 workers)
   Custom prefix: Auto-detect
   Generate full sample: True
   Time match: True
   Generate chords: True
   Pitch shift method: both
   Chord qualities: Major, Minor
   Selected specific chord types:
     Major: Major fifth, Major seventh
     Minor: Minor fifth, Minor seventh
   Generate inversions: True
   Selected inversions for Major seventh: 1st, 2nd
   Play notes: False
   Overwrite existing: False
   Keep artifacts: False
   
   Proceed with these settings? (y/n)
   ```
   
   - Review the settings and confirm to proceed
   - If you select 'n', the process will be canceled

8. **Processing**

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

## Changelog

### v1.1
- Renamed from "note_expander" to "Pitchbinder"
- Completely redesigned chord generation interface
- Added granular control over chord types and inversions:
  - Select specific chord qualities (Major, Minor, etc.)
  - Select specific chord types within each quality
  - Select specific inversions for each chord type
- Improved organization of generated chords
- Optimized processing workflow