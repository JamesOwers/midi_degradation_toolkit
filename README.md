# mdtk - The MIDI Degradation Toolkit
Tools to generate datasets of Altered and Corrupted MIDI Excerpts -`ACME`
datasets.

The accompanying paper (submitted to ISMIR)
"Symbolic Music Correction using The MIDI Degradation Toolkit" describes the
toolkit and its motivation in detail. For instructions to reproduce the results
from the paper, see [`./baselines/README.md`](./baselines/README.md).

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
* `onset` is the time in milliseconds when a note began,
* `track` is the identifier for a distinct track in the midi file,
* `pitch` is the midinote pitch number ranging from 0 (C-2) to 127 (G9) (concert
  A4 is midinote 69), and
* `dur` is how long the note is held in milliseconds.

There are then functions to alter these files, introducing un-musical
degradations such as pitch shifts.

Finally, the toolkit also contains modules to aid modelling, such as pytorch
dataset classes for easy data loading.

Some highlights include:
* [`mdtk.downloaders`](./mdtk/downloaders.py) - classes used to download midi
  data for use
* [`mdtk.degradations`](./mdtk/degradations.py) - functions to alter midi data
  e.g. `pitch_shift` or `time_shift`
* [`mdtk.degrader`](./mdtk/degrader.py) - Degrader class that can be used to
  degrade data points randomly on the fly
* [`mdtk.eval`](./mdtk/eval.py) - functions for evaluating model performance
  on each task, given a list of outputs and targets
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

For usage instructions for the `measure_errors.py` script, run
`python measure_errors.py -h` you should create a directory of transcriptions
and a directory of ground truth files (in mid or csv format). The ground truth
and corresponding transcription should be named the exact same thing.

See `measure_errors_example.ipynb` for an example of the script's usage.


## Contributors
If you would like to contribute, please install in developer mode and use the dev option
when installing the package. Additionally, please run `pre-commit install` to
automatically run pre-commit hooks.

```bash
pip install -e ${path_to_repo}[dev]
pre-commit install
```
