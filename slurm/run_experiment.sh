 #!/usr/bin/env bash
EXPT_FILE=$1
if [ $# == 1 ]; then
    echo 'No max nr of parallel expts provided, setting to 10'
    MAX_PARALLEL_JOBS=10 
else
    MAX_PARALLEL_JOBS=$2 
fi
SBATCH_ARGS="${@:3}"
NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
echo "Executing command: sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} $SBATCH_ARGS slurm_arrayjob.sh $EXPT_FILE"
sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} $SBATCH_ARGS slurm_arrayjob.sh $EXPT_FILE