Add the scripts in the folder to the path so they can reference one another
regardless of where you are running the experiment from.

# Setup
```
project_home=/home/$USER/git/midi_degradation_toolkit
cd $project_home/slurm
chmod u+x {run_experiment.sh,slurm_arrayjob.sh,gen_task*}
echo "export PATH=/home/$USER/git/midi_degradation_toolkit/slurm:\$PATH" >> ~/.bashrc
source ~/.bashrc
```

# Run an experiment
For example, to run the experiment for task 1:

```
# Make the experiment file
gen_task1_expt.py
expt_file=$project_home/slurm/experiments/task1_experiments.txt
# Run the experiment
max_nr_concurrent_jobs=8
run_experiment.sh $expt_file $max_nr_concurrent_jobs
```
