#!/usr/bin/env python
import argparse
import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mdtk.pytorch_models
import mdtk.pytorch_trainers
from mdtk.pytorch_datasets import transform_to_torchtensor
from mdtk.formatters import CommandVocab, FORMATTERS, create_corpus_csvs
from mdtk.eval import helpfulness
from mdtk.degradations import MIN_PITCH_DEFAULT, MAX_PITCH_DEFAULT




## For dev mode warnings...
#import sys
#if not sys.warnoptions:
#    import warnings
#    warnings.simplefilter("always") # Change the filter in this process
#    os.environ["PYTHONWARNINGS"] = "always" # Also affect subprocesses


# For user mode warnings...
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore") # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    


# TODO: get formatter out of Trainer
# TODO: remove eval arg from Trainer iteration method and do eval outside
def construct_parser():
    parser = argparse.ArgumentParser()

    # Filepath stuff
    parser.add_argument("-i", "--input", default='acme', help='The '
                        'base directory of the ACME dataset to use as input.')
    parser.add_argument("-m", "--model", required=True, help='The '
                        'filename of the model to use in evaluation.')

    # Basic task setup args
    parser.add_argument("--format", required=False, choices=FORMATTERS.keys(),
                        help='The format to use as input to the model. If the '
                        'format-specific csvs have not yet been created, this '
                        'will create them. Choices are '
                        f'{list(FORMATTERS.keys())}. Required if --baseline '
                        'is not given.')

    parser.add_argument("-t", "--task", required=True, choices=range(1, 5), help='The '
                        'task number to train a model for.', type=int)

    parser.add_argument("--baseline", action='store_true', help='Ignore all '
                        'arguments and run the baseline model for the given '
                        '--task.')
    
    # Dataset structure arg
    parser.add_argument("-s", "--seq_len", type=int, default=250,
                        help="maximum sequence length.")

    # Training/DataLoading args
    parser.add_argument("--splits", nargs='+', default=['test'],
                        help="which splits to evaluate: train, valid, test.")
    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="number of batch_size")
    parser.add_argument("-w", "--num_workers", type=int, default=4,
                        help="dataloader worker size")
    parser.add_argument("--with_cpu", action='store_true', default=False,
                        help="Train with CPU, default is to try and use CUDA. "
                        "A warning will be thrown if CUDA is not available, "
                        "and CPU used in that case.")
    parser.add_argument("--cuda_devices", type=int, nargs='+',
                        default=None, help="CUDA device ids")
    parser.add_argument("--in_memory", type=bool, default=True,
                        help="Loading on memory: true or false")

    # Piano-roll specific size args
    parser.add_argument("--pr-min-pitch", type=int, default=MIN_PITCH_DEFAULT,
                        help="Minimum pianoroll pitch")
    parser.add_argument("--pr-max-pitch", type=int, default=MAX_PITCH_DEFAULT,
                        help="Maximum pianoroll pitch")
    return parser


def main(args):
    if args.baseline:
        # Setup args for the baseline for args.task
        raise NotImplementedError(f"Baseline not created for task {args.task} yet.")
    else:
        assert args.format is not None, (
            '--format is a required argument if --baseline is not given.'
        )

    # Generate (if needed) and load formatted test csv
    prefix = FORMATTERS[args.format]["prefix"]
    if not os.path.exists(os.path.join(args.input,
                                       f'test_{prefix}_corpus.csv')):
        create_corpus_csvs(args.input, FORMATTERS[args.format])
    dataset_path = {
        'train': os.path.join(args.input, f'train_{prefix}_corpus.csv'),
        'valid': os.path.join(args.input, f'valid_{prefix}_corpus.csv'),
        'test': os.path.join(args.input, f'test_{prefix}_corpus.csv')
    }
    
    task_idx = args.task - 1
    task_name = task_names[task_idx]
    model_name = FORMATTERS[args.format]['models'][task_idx]
    if model_name is None:
        raise NotImplementedError("No model implemented to load for task "
                                  f"{task_name} with format {args.format}")
    Dataset = getattr(mdtk.pytorch_datasets, FORMATTERS[args.format]['dataset'])
    Trainer = task_trainers[task_idx]
    Criterion = task_criteria[task_idx]
    
    if args.format == 'command':
        vocab = CommandVocab()
        dataset_args = [vocab, args.seq_len]
        dataset_kwargs = {
        }
    elif args.format == 'pianoroll':
        dataset_args = [args.seq_len]
        dataset_kwargs = {
            'min_pitch': args.pr_min_pitch,
            'max_pitch': args.pr_max_pitch
        }
    
    dataset = {}
    dataloader = {}
    for split in args.splits:
        path = dataset_path[split]
        print(f"Loading {split} {Dataset.__name__} from {path}")
        
        dataset[split] = Dataset(path, *dataset_args, **dataset_kwargs,
                                 in_memory=args.in_memory,
                                 transform=transform_to_torchtensor)

        print(f"Creating {split} DataLoader")
        dataloader[split] = DataLoader(dataset[split], batch_size=args.batch_size,
                                       num_workers=args.num_workers)
    
    print(f"Loading model {args.model}")
    model = torch.load(args.model, map_location='cpu')
        
    print("Creating Tester")
    with_cuda = not args.with_cpu
    if with_cuda:
        print("Attempting to test on GPU")
    else:
        print("Attempting to test on CPU")
    
    split_log_info = {}
    for split in args.splits:
        trainer = Trainer(
            model=model,
            criterion=Criterion,
            train_dataloader=None,
            test_dataloader=dataloader[split],
            with_cuda=with_cuda,
            batch_log_freq=None,
            epoch_log_freq=None,
            formatter=FORMATTERS[args.format],
            log_file=None
        )
        print(f"Evaluating {split} split")
        log_info = trainer.test(0, evaluate=True)
        split_log_info[split] = log_info
    return split_log_info


task_names = [
    'ErrorDetection',
    'ErrorClassification',
    'ErrorLocation',
    'ErrorCorrection'
]

task_trainers = [
    getattr(mdtk.pytorch_trainers, f'{task_name}Trainer')
    for task_name in task_names
]

task_criteria = [
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.BCEWithLogitsLoss(reduction='mean')
]



if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
