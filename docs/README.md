# MDTK Documentation

This directory contains documentation with examples for the toolkit. This includes:


1. [The ACME dataset](./01_the_ACME_dataset.ipynb)
    * an introduction and
    * a description of the dataset used in our paper
2. [Dataset creation](./02_dataset_creation.ipynb)
    * how to create your own ACME datasets
3. [Degradation functions](./03_degradation_functions.ipynb)
    * an introduction to the available functions, and
    * their parameters
4. [Data providers and the degrader class](./04_data_provider_and_degrader.ipynb)
    * How to provide data to, for example, pytorch models
    * How to augment a dataset with degradations on-the-fly
5. [Matching errors with your AMT system](./05_AMT_error_matching.ipynb)
    * How to generate data which matches the output of your AMT system
6. [Reproducing results from the paper](./06_training_and_evalutation.ipynb)
    * loads the fitted parameters
    * describes the models
    * outlines the training procedure
    * describes evaluation
    * script to perform training & evaluation to reproduce paper results provided

## Augmenting your AMT system
If you are interested in augmenting the data to train a model which cleans the output
of **your AMT system**:

1. Read [how to match the errors of your AMT system](./05_AMT_error_matching.ipynb), then
    either
    1. how to [create a dataset](./01_dataset_creation.ipynb) with parameters to match
        your AMT system's errors, or
    2. [degrade data on the fly](./04_data_provider_and_degrader.ipynb)
2. [Train a model](./06_training_and_evalutation.ipynb) on the degraded data to fix the errors
