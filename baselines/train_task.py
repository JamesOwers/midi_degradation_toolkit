#!/usr/bin/env python
# TODO: turn this into a script which can train any of the tasks given 
# a command line argument
import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import mdtk.pytorch_models
import mdtk.pytorch_trainers
from mdtk.pytorch_datasets import (transform_to_torchtensor,
                                   CommandDataset, PianorollDataset)
from mdtk.formatters import CommandVocab, FORMATTERS, create_corpus_csvs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", default='acme', help='The '
                        'base directory of the ACME dataset to use as input.')
    parser.add_argument("--format", required=True, choices=FORMATTERS.keys(),
                        help='The format to use as input to the model. If the '
                        'format-specific csvs have not yet been created, this '
                        'will create them. Choices are '
                        f'{list(FORMATTERS.keys())}')

    parser.add_argument("-o", "--output_path", required=False, type=str, help="ex)output/bert.model")

    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()
    return args



task_names = [
    'ErrorDetection',
    'ErrorClassification',
    'ErrorIdentification',
    'ErrorCorrection'
]

task_models = [
    getattr(mdtk.pytorch_models, f'{task_name}Net')
    for task_name in task_names
]

task_trainers = [
    getattr(mdtk.pytorch_trainers, f'{task_name}Trainer')
    for task_name in task_names
]

task_datasets = [
    CommandDataset,
    CommandDataset,
    CommandDataset,
    CommandDataset
]

task_criteria = [
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss()
]



if __name__ == '__main__':
    
    # TODO: Actually do proper arg parsing as opposed to hacking it >_<
    base_dir = '.'
    args = parse_args()

    # Generate (if needed) and load formatted csv
    prefix = FORMATTERS[args.format]["prefix"]
    if not all([os.path.exists(
            os.path.join(args.input, f'{split}_{prefix}_corpus.csv')
        ) for split in ['train', 'valid', 'test']]):
        create_corpus_csvs(args.input, FORMATTERS[args.format])
    args.train_dataset = os.path.join(args.input, f'train_{prefix}_corpus.csv')
    args.valid_dataset = os.path.join(args.input, f'valid_{prefix}_corpus.csv')
    args.test_dataset = os.path.join(args.input, f'test_{prefix}_corpus.csv')

    # TODO: decide on reasonable length
    args.seq_len = 100
    args.in_memory = True
    args.batch_size = 64
    args.num_workers = 4
    args.task_number = 1
    args.lr = 1e-4
    args.b1 = 0.9
    args.b2 = 0.999
    args.weight_decay = 0.01
    args.with_cuda = False
    args.batch_log_freq = 10
    args.epoch_log_freq = 1
    args.epochs = 1000
    args.output_path = f'{base_dir}/model.checkpoint'
    
    
    task_idx = args.task_number - 1
    task_name = task_names[task_idx]
    Model = task_models[task_idx]
    Trainer = task_trainers[task_idx]
    Dataset = task_datasets[task_idx]
    Criterion = task_criteria[task_idx]
    
    print("Loading Vocab")
    vocab = CommandVocab()
    vocab_size = len(vocab)
    print("Vocab Size: ", vocab_size)

    print(f"Loading train {Dataset.__name__} from {args.train_dataset}")
    train_dataset = Dataset(args.train_dataset, vocab,
                            seq_len=args.seq_len,
                            in_memory=args.in_memory,
                            transform=transform_to_torchtensor)

    print(f"Loading test {Dataset.__name__} from {args.test_dataset}")
    test_dataset = Dataset(args.test_dataset, vocab,
                           seq_len=args.seq_len,
                           in_memory=args.in_memory,
                            transform=transform_to_torchtensor)

    print(f"Creating train and test DataLoaders")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    print(f"Building {Model.__name__}")
    model = Model(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=100,
        output_size=2,
        dropout_prob=0.1
    )
    
    print(f"Using {Criterion.__str__()} as loss function")
    
    print("Creating Trainer")
    trainer = Trainer(
        model=model,
        criterion=Criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        lr=args.lr,
        betas=(args.b1, args.b2),
        weight_decay=args.weight_decay,
        with_cuda=args.with_cuda,
        batch_log_freq=args.batch_log_freq,
        epoch_log_freq=args.epoch_log_freq
    )
    
    print("Training Start")
    for epoch in range(args.epochs):
        # I test before train as then both train and test values are using
        # the same set of parameters for the same epoch number
        if test_dataloader is not None:
            trainer.test(epoch)
        
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)