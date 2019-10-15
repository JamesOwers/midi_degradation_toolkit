#!/usr/bin/env python
import os

code_dir = os.path.expanduser('~/git/melody_gen')
out_dir = '/mnt/cdtds_cluster_home/s0816700/git/melody_gen/data/output'
in_dir = '/disk/scratch/s0816700/data/mirex_p4p'

base_call = (
    f"python train.py --data-home {in_dir} "
    f"-t PPDD-Jul2018_sym_mono_large -v PPDD-Jul2018_sym_mono_medium "
    f"--valid-dataset-cache-loc "
    f"{in_dir}/working/PPDD-Jul2018_sym_mono_medium.h5 "
    f"--num-epochs 1000 "
    f"--batch-size 64 "  # TODO: either raise this or lower lr in future
    f"--device best "
    f"--early-stopping 25 "
    f"-o {out_dir} "
    f"--no-timestamp ")

nr_repeats = 10
learning_rates = [0.001, 0.0001, 0.00001]
models = ['ConvNet3', 'ConvNet4', 'ConvNet5']
mults = [1, 2, 3]
transforms = ['', '--transform transpose_beta_binomial']
nr_expts = (nr_repeats * len(learning_rates) * len(models) * len(mults) * 
            len(transforms))

nr_servers = 10
avg_expt_time = 60  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

settings = [(transform, model, mult, lr, repeat) for transform in transforms
            for model in models for mult in mults for lr in learning_rates 
            for repeat in range(nr_repeats)]

high_level_expt_name = 'mdl5'
output_file = open(f"{code_dir}/scripts/{high_level_expt_name}_experiments.txt", "w")
for transform, model, mult, lr, repeat in settings:
    data_aug = '_data-aug' if transform else '_no-data-aug'
    train_cache = (f"{in_dir}/working/"
                   f"PPDD-Jul2018_sym_mono_large{data_aug}.h5")
    expt_name = f"{high_level_expt_name}__{model}_{mult}_{lr}{data_aug}_{repeat}"
    model_call = f"'{model}(mult={mult})'"
    # Will be randomly seeded, and this seed is reported in logs
    expt_call = (f"{base_call} "
                 f"--expt-name {expt_name} "
                 f"--model {model_call} "
                 f"--learning-rate {lr} "
                 f"{transform} "
                 f"--train-dataset-cache-loc {train_cache} "
                 f"--checkpoint {out_dir}/{expt_name}/checkpoint.tar")
    print(expt_call, file=output_file)

output_file.close()