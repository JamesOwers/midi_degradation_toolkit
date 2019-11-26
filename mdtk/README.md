# Description

The accompanying paper (submitted to ICASSP, available upon request) referenced
below is "Symbolic Music Correction using The MIDI Degradation Toolkit" and 
describes the toolkit and its motivation in detail.

* `data_structures.py` - contains functions for reading data and the main data
  structure class we will use for storing clean data
* `degradations.py` - code to perform the degradations i.e. edits to the midi
  data e.g. pitch shifting
* `downloaders.py` - code to download each datasets and convert to the standard
  format. If you would like to contribute a new dataset, add a new class to
  this file with an appropreate name and ensure it has a method `download_midi`
  which puts all midi files in a specified directory `output_path`)
* `eval.py` - functions for calculating evaluation metrics for each task
  outlined in the paper, taking as input a list of outputs and targets as
  required
* `filesystem_utils.py` - utility funcitons for moving and copying files etc
* `formatters.py` - functions for conversion of data, especially for creating
  the much smaller single files read by the pytorch_dataset classes
* `midi.py` - conversion of midi files into other formats
* `pytorch_datasets.py` - classes to use which return data items by index
* `pytorch_models.py` - the baseline models we propose in the paper
* `pytorch_trainers.py` - classes which coordinate the training of the models
