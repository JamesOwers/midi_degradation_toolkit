# mdtk - The MIDI Degradataion Toolkit
Tools to generate datasets of Altered and Corrupted MIDI Excerpts - `ACME` datasets.

## Install

```
git clone https://github.com/JamesOwers/midi_degradation_toolkit
cd midi_degradation_toolkit
pip install ./  # use pip install -e ./ for dev mode if you want to edit files
```

## Quickstart

To generate an `ACME` dataset simply install the package i.e. `pip install mdtk` (or as above).

* By default this will download a small test dataset found in `./data`
* Pytorch datasets for use with pytorch dataloaders are located in `mdtk.pytorch_datasets`
* To install more pre-configured datasets, or change this directory data are downloaded to, see the documentation for 
`./scripts/gen_acme_dataset.py`
* To download new datasets, install from source, and add new configurations to `mdtk.download`


