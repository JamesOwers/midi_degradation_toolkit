# mdtk - The MIDI Degradation Toolkit
Tools to generate datasets of Altered and Corrupted MIDI Excerpts -`ACME`
datasets.

The accompanying paper (submitted to ICASSP, available upon request)
"Symbolic Music Correction using The MIDI Degradation Toolkit" describes the
toolkit and its motivation in detail.

The script for creating datasets is [`./make_dataset.py`](make_dataset.py).
However, components in the tookit can be used standalone - for example the
degradation functions, and pytorch dataset classes.

As a brief overview, the toolkit takes midi files as input and first converts
them to a standard data structure like this:
```
onset,track,pitch,dur
0    ,0    ,100  ,250
250  ,0    ,105  ,255
250  ,1    ,100  ,100
...
```
where:
* onset is the time in milliseconds when a note began,
* track is the identifier for a distinct track in the midi file,
* pitch is the midinote pitch number ranging from 0 (C-2) to 127 (G9) (concert 
  A4 is midinote 69),
* and dur is how long the note is held in milliseconds.

There are then functions to alter these files, introducing un-musical
degradations such as pitch shifts.

Finally, the toolkit also contains modules to aid modelling, such as pytorch
dataset classes for easy data loading.

Some highlights include:
* [`mdtk.downloaders`](./mdtk/downloaders.py) - classes used to download midi
  data for use
* [`mdtk.degradations`](./mdtk/degradations.py) - functions to alter midi data
  e.g. `pitch_shift` or `time_shift`
* [`mdtk.formatters`](./mdtk/formatters.py) - functions converting between
  different data formats, e.g. making very small files for pytorch dataset
  classes to read
* [`mdtk.pytorch_datasets`](./mdtk/pytorch_datasets.py) - pytorch dataset
  classes for quickly loading and modelling data (not restricted to pytorch
  models!)
* [`./baselines`](./baselines) - scripts for running the baseline models
  included in the paper (available upon request)

For more information about mdtk modules, see the package readme: 
[`./mdtk/README.md`](./mdtk/README.md)

## Install
We recommend using an enviroment manager such as conda, but you may omit these
lines if you use something else. This install will allow you to both run all
the scripts in this repository and use the toolkit in your own scripts
(`import mdtk`). The only requirement is **python version 3.7**.

```
git clone https://github.com/JamesOwers/midi_degradation_toolkit
cd midi_degradation_toolkit
conda update conda
conda create -n mdtk python=3.7
conda activate mdtk
pip install .  # use pip install -e . for dev mode if you want to edit files
```

## Quickstart

To generate an `ACME` dataset simply install the package with instructions
above and run `./make_dataset.py`.
