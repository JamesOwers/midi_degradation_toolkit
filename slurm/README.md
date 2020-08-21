Add the scripts in the folder to the path so they can reference one another
regardless of where you are running the experiment from.

# Setup
Install the cluster scripts by following:
* https://github.com/JamesOwers/cluster-scripts#setup, and
* https://github.com/JamesOwers/cluster-scripts/tree/master/experiments#setup


# Generate commands, and run an experiment
For example, to run the experiment for task 1:

```bash
project_home=$HOME/git/midi_degradation_toolkit

# Make the experiment file
cd $project_home/slurm
# expt_name=task1
# expt_name=task1weighted
expt_name=task2
# expt_name=task3
# expt_name=task3weighted
# expt_name=task4
python gen_${expt_name}_expt.py
expt_file=$project_home/slurm/experiments/${expt_name}_experiments.txt

# Run the experiment
max_nr_concurrent_jobs=9
run_experiment \
    -b slurm_arrayjob.sh \
    -e $expt_file \
    -m $max_nr_concurrent_jobs
```

# Tips

Jupyter notebooks

Expose port:
```bash
local_port=9999
remote_port=8888
user=s0816700
host=mlp.inf.ed.ac.uk
#host=cdtcluster.inf.ed.ac.uk
ssh -L ${local_port}:localhost:${remote_port} ${user}@${host}
```

Spin up the server on remote (not on headnode, probably in tmux):
```bash
jupyter lab --no-browser --port=8888 --NotebookApp.token= &
```
