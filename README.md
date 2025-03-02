# Note Expander

A powerful tool for expanding a small set of piano samples into a complete 8-octave range.

## Overview

Note Expander takes a few piano samples with note information in their filenames and intelligently generates the missing notes across the full piano range (C1-B8). It uses high-quality pitch shifting and optional time stretching to create natural-sounding samples for all 88 piano keys.

## Features

- **Intelligent Sample Generation**: Creates missing notes based on your existing samples.
- **Filename Detection**: Automatically detects note and octave from filenames.
- **Time Matching**: Optionally matches all generated samples to the average length of source samples.
- **Full Sample Generation**: Creates a single WAV file with all notes in sequence.
- **Slice Marking**: Embeds full sample with markers for each note, compatible with Dirtywave M8 and more.
- **Recursive Processing**: Process multiple directories at once.
- **Playback**: Option to play all generated notes in sequence after processing.

## How It Works

Note Expander analyzes your existing samples, identifies the notes they represent, and uses pitch-shifting algorithms to generate the missing notes. It finds the closest existing sample to use as a source for each new note, ensuring the most natural sound possible.

### Supported Filename Formats

The script can detect notes and octaves from filenames in formats like:
- `Piano-C4.wav`
- `Grand Piano-F#3.wav`
- `MyPiano-G#5.wav`
- `CP70-A2.wav`

As long as the filename contains a note letter (A-G), optional sharp (#), and octave number, it will be detected.

## Installation

1. Clone this repository or download the script
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Basic usage:
```
python note_expander.py [options] [source_directory]
```

If no source directory is specified, the current directory is used.

### Command Line Options

- `-r, --recurse`: Process all subdirectories recursively
- `-p PREFIX, --prefix PREFIX`: Prefix for generated files (default: auto-detect)
- `-P, --play`: Play all notes when done
- `-f, --gen-full`: Generate a single WAV file with all notes in sequence
- `-t, --time-match`: Match all generated samples to the average length of source samples
- `-o, --overwrite`: Overwrite existing expansion directories and regenerate all samples

### Examples

Process the current directory:

```
python note_expander.py
```

Process a specific directory with time matching:
```
python note_expander.py -t /path/to/samples
```

Generate a full sample file with all notes:
```
python note_expander.py -f /path/to/samples
```

Process all subdirectories recursively:
```
python note_expander.py -r /path/to/parent/directory
```

Full processing with all processing options:
```
python note_expander.py -fort /path/to/samples
```

## Output

All generated samples are saved to an `expansion` subdirectory within each processed directory. This keeps your original samples untouched while organizing the new ones.

When using the `-f` option, a single WAV file containing all notes in sequence is generated, along with a CUE file that marks the position of each note.

## Tips for Best Results

1. **Use High-Quality Source Samples**: The better your source samples, the better the generated ones will be.
2. **Provide Samples Across the Range**: For best results, provide samples that span the range (e.g., one sample per octave).
3. **Consistent Naming**: Make sure your sample filenames follow a consistent pattern with clear note information.
4. **Time Matching**: Use the `-t` option if you want all samples to have the same duration.

## Limitations

- The quality of pitch-shifted samples decreases as they get further from the source sample.
- Very extreme pitch shifts (e.g., shifting a C1 to a C8) may not sound natural.