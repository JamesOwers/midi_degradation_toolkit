# Baselines

This directory contains scripts which we used to fit and analyse the results
from all the baseline models proposed in the accompanying paper (submitted to
ICASSP, available upon request) "Symbolic Music Correction using The MIDI
Degradation Toolkit".

Follow the instructions below to reproduce all models, figures, and tables from
the paper.

## Reproducing results
1. Download ACME v1.0 dataset
    * Go to https://github.com/JamesOwers/acme and extract the zip into
    the root directory of the repo
    * alternatively run `./make_dataset --seed 1762218506` with the repo
    set at the tag ACMEv1.0
    * you should now have the directory structure:
    ```
    midi_degradation_toolkit
        mdtk
        baselines
        ...
        acme
            altered
            clean
            ...
            valid_pr_corpus.csv
    ```
1. Fit all the models
    * To fit all the models, run all of the commands contained within the file
    [`experiment.txt`](./experiment.txt). These are all python commands which
    expect you to have installed the package.
    * we recommend running from the base directory and saving the outputs to
    a folder there. For example, to run the first 5 experiments:
    ```
    repo_dir=..
    cd $repo_dir
    head baselines/experiment.txt | bash
    ```
    * You should now have a folder `$repo_dir/output` which containing
    subdirectories with the name of each task
    * in each subdirectory, there are model checkpoint files and the
    training log
    * Tip: if you want to save the results somewhere else, edit
    `experiment.txt` using `sed`:
    ```
    sed -i 's_output_some/other/place_g' baselines/experiment.txt
    ```
    * Tip: we used distributed computing with `slurm` (approx 20 experiments at
    a time) to run our experiments in parallel. Each node had a GPU. In this
    setting, it took about 1 day to run all the experiments in
    `experiment.txt`
1. Produce plots and tables from experiments that have been run by running
[`./baselines/get_results.py`]:
    * Assuming you now have an directory structure as outlined above
    and you ran all the experiments in `experiment.txt`, you can run
    the following command to produce all the plots and results tables
    ```
    python ./baselines/get_results.py --output_dir output --save_plots output --in_dir acme --task_names task1 task1weighted task2 task3 task3weighted task4 --setting_names "['lr','wd','hid']" "['lr','wd','hid']" "['lr','wd','hid']" "['lr','wd','hid']" "['lr','wd','hid']" "['lr','wd','hid','lay']" --formats command command command pianoroll pianoroll pianoroll --seq_len 1000 1000 1000 250 250 250 --metrics rev_f rev_f avg_acc f f helpfulness --task_desc ErrorDetection ErrorDetection ErrorClassification ErrorLocation ErrorLocation ErrorCorrection --splits train valid test
    ```
    * This will load the models and perform evaluation, so it is again
    recommended that you execute the above command with GPU availability
1. Observe the results!
    * 
    