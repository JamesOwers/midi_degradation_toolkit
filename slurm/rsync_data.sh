#!/usr/bin/env sh
## Template for use with a file containing a list of commands to run
## Example call:
##     EXPT_FILE=scripts/experiments.txt
##     NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
##     MAX_PARALLEL_JOBS=4 
##     sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_template.sh $EXPT_FILE

#SBATCH -o /mnt/cdtds_cluster_home/s0816700/slurm_logs/slurm-%A_%a.out
#SBATCH -e /mnt/cdtds_cluster_home/s0816700/slurm_logs/slurm-%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --mem=14000  # memory in Mb
#SBATCH -t 2:30:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.
#SBATCH --mail-user=james.owers@ed.ac.uk
#SBATCH --mail-type=ALL
# #SBATCH --exclude=charles15  # Had an outdated nvidia driver, fixed now


set -e  # make script bail out after first error


# slurm info for logging
echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo $dt


# Set environment variables for input and output data
echo 'Setting experiment environment variables'
export STUDENT_ID=$(whoami)
export SCRATCH_HOME=/disk/scratch/${STUDENT_ID}
mkdir -p ${SCRATCH_HOME}
export TMPDIR=${SCRATCH_HOME}
export TMP=${SCRATCH_HOME}
mkdir -p ${DATA_HOME}
export CLUSTER_HOME=/mnt/cdtds_cluster_home/${STUDENT_ID}

code_repo_home=${CLUSTER_HOME}/git/midi_degradation_toolkit
output_dir=${code_repo_home}/output
distfs_data_home="${code_repo_home}/acme"
scratch_data_home="${SCRATCH_HOME}/data/acme"


# Move data from distrbuted filesystem to scratch dir on cluster node
for corpus in cmd pr; do
    for split in train valid test; do
        fn="${split}_${corpus}_corpus.csv"
        source="${distfs_data_home}/${fn}"
        target="${scratch_data_home}/${fn}"
        rsync -ua --progress ${source} ${target}
    done
done

for fn in degradation_ids.csv labels.csv metadata.csv; do
    source="${distfs_data_home}/${fn}"
    target="${scratch_data_home}/${fn}"
    rsync -ua --progress ${source} ${target}
done


echo "============"
echo "job finished successfully"