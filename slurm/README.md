Add the scripts in the folder to the path so they can reference one another
regardless of where you are running the experiment from.

chmod u+x {run_experiment.sh,slurm_arrayjob.sh}
echo "export PATH=/home/$USER/git/midi_degradation_toolkit/slurm:\$PATH" >> ~/.bashrc
source ~/.bashrc