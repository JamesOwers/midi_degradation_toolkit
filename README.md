[build-image]: https://travis-ci.com/JamesOwers/midi_degradation_toolkit.svg?branch=master
[build-url]: https://travis-ci.com/JamesOwers/midi_degradation_toolkit
[coverage-image]: https://codecov.io/gh/JamesOwers/midi_degradation_toolkit/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/JamesOwers/midi_degradation_toolkit?branch=master
[license-image]: https://img.shields.io/github/license/JamesOwers/midi_degradation_toolkit
[license-url]: https://github.com/JamesOwers/midi_degradation_toolkit/blob/master/LICENSE
[arxiv-image]: http://img.shields.io/badge/cs.SD-arXiv%3A2010.00059-B31B1B.svg
[arxiv-url]: https://arxiv.org/abs/2010.00059
<!-- [docs-image]: https://readthedocs.org/projects/midi_degradation_toolkit/badge/?version=latest
[docs-url]: https://midi_degradation_toolkit.readthedocs.io/en/latest/?badge=latest
[pypi-image]: https://badge.fury.io/py/midi_degradation_toolkit.svg
[pypi-url]: https://pypi.python.org/pypi/midi_degradation_toolkit -->

<p align="center">
  <img width="90%" src="https://raw.githubusercontent.com/JamesOwers/midi_degradation_toolkit/master/img/mdtk_logo.png?sanitize=true" />
</p>

----------------------------------------------------------------------------------------

# mdtk - The MIDI Degradation Toolkit

[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]
[![GitHub license][license-image]][license-url]
[![ArXiv Paper][arxiv-image]][arxiv-url]
<!-- [![PyPI Version][pypi-image]][pypi-url] -->
<!-- [![Docs Status][docs-image]][docs-url] -->

Tools to generate datasets of Altered and Corrupted MIDI Excerpts -`ACME`
datasets. Baseline models for cleaning the output from Automatic Music Transcription systems.

The accompanying paper, The MIDI Degradation Toolkit: Symbolic Music Augmentation and Correction, describes the
toolkit and its motivation in detail. For instructions to reproduce the results
from the paper, see the documentation [`./docs/06_training_and_evaluation.ipynb`](./docs/06_training_and_evaluation.ipynb).

## Documentation
Documentation for the components of the toolkit is provided in [`./docs`](./docs)

## Overview
As a brief overview, the toolkit takes midi files as input and first converts
them to a standard data structure like this:

onset|track|pitch|dur|velocity
---|---|---|---|---
0|0|100|250|80
250|0|105|255|100
250|1|100|100|95

where:
* `onset` is the time in milliseconds when a note began,
* `track` is the identifier for a distinct track in the midi file,
* `pitch` is the midinote pitch number ranging from 0 (C-2) to 127 (G9) (concert
  A4 is midinote 69), and
* `dur` is how long the note is held in milliseconds.
* `velocity` is the velocity of the note (defaults to 100 if not parsing from MIDI).

There are then functions to alter these files, introducing un-musical
degradations such as pitch shifts.

The toolkit also contains modules to aid modelling, such as pytorch
dataset classes for easy data loading.

For a more comprehensive overview, see the documentation avaialbe in [`./docs`](./docs)

## Install
We recommend using an environment manager such as conda, but you may omit these
lines if you use something else. This install will allow you to both run all
the scripts in this repository and use the toolkit in your own scripts
(`import mdtk`). The requirements are described in the `setup.cfg` file.

```bash
git clone https://github.com/JamesOwers/midi_degradation_toolkit
cd midi_degradation_toolkit
conda update conda
conda create -n mdtk python=3.7
conda activate mdtk
pip install .
```

There are install options available:
```bash
# use pip install -e . for dev mode if you want to edit files
pip install -e ".[dev]"  # install everything
pip install -e ".[docs]"  # packages to rerun documentation
pip install -e ".[eval]"  # required for reproducing results from paper
```

## Quickstart
To generate an `ACME` dataset simply install the package with instructions
above and run `python make_dataset.py`.

For usage instructions for the `measure_errors.py` script, run
`python measure_errors.py -h` you should create a directory of transcriptions
and a directory of ground truth files (in mid or csv format). The ground truth
and corresponding transcription should be named the exact same thing.

Training and evaluation code for the proposed modelling tasks is contained in [`./baselines`](./baselines)

## Contributors
If you would like to contribute, please install in developer mode and use the dev option
when installing the package. Additionally, please run `pre-commit install` to
automatically run pre-commit hooks.

```bash
pip install -e "${path_to_repo}[dev]"
pre-commit install
```
