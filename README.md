# mdtk - The MIDI Degradation Toolkit
Tools to generate datasets of Altered and Corrupted MIDI Excerpts -`ACME`
datasets.

## Install
We recommend using an enviroment manager such as conda, but you may omit these
lines if you use something else. This install will allow you to `import mdtk`
and run all the scripts in this repository. The only requirement is python
version 3.7.

```
git clone https://github.com/JamesOwers/midi_degradation_toolkit
cd midi_degradation_toolkit
conda create -n mdtk python=3.7
conda activate mdtk
pip install .  # use pip install -e . for dev mode if you want to edit files
```

## Quickstart

To generate an `ACME` dataset simply install the package with instructions
above and run `./make_dataset.py`.

