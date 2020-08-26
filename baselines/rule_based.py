#!/usr/bin/env python

import argparse
import os

import numpy as np
import torch.nn

import mdtk.pytorch_datasets
from mdtk.formatters import FORMATTERS, create_corpus_csvs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        default="acme",
        help="The " "base directory of the ACME dataset to use as input.",
    )
    parser.add_argument(
        "-s",
        "--seq_len",
        type=int,
        default=250,
        help="maximum sequence length for a pianoroll.",
    )
    parser.add_argument(
        "--reformat",
        action="store_true",
        help="Force the "
        "creation of the pianoroll csvs, even if they "
        "already exist.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    prefix = FORMATTERS["pianoroll"]["prefix"]
    if (
        not all(
            [
                os.path.exists(os.path.join(args.input, f"{split}_{prefix}_corpus.csv"))
                for split in ["train", "test"]
            ]
        )
    ) or args.reformat:
        create_corpus_csvs(args.input, FORMATTERS["pianoroll"])
    train_dataset = os.path.join(args.input, f"train_{prefix}_corpus.csv")
    test_dataset = os.path.join(args.input, f"test_{prefix}_corpus.csv")

    # Calculate outputs
    seq_len = args.seq_len
    dataset = mdtk.pytorch_datasets.PianorollDataset(train_dataset, seq_len)
    deg_counts = np.zeros(9)
    frame_counts = np.zeros(2)
    pr_outputs = np.zeros((2, 2))
    for data in dataset:
        deg_counts[data["deg_label"]] += 1
        sum_frames = np.sum(data["changed_frames"])
        frame_counts[0] += seq_len - sum_frames
        frame_counts[1] += sum_frames
        deg_pr = data["deg_pr"]
        clean_pr = data["clean_pr"]
        num_pitches = np.shape(clean_pr)[1]
        for deg in [0, 1]:
            for clean in [0, 1]:
                pr_outputs[clean, deg] += np.sum(
                    np.where(np.logical_and(deg_pr == deg, clean_pr == clean), 1, 0)
                )
    deg_counts /= np.sum(deg_counts)
    frame_counts /= np.sum(frame_counts)
    pr_outputs /= np.sum(pr_outputs, axis=0)

    # Calculate losses and metrics
    dataset = mdtk.pytorch_datasets.PianorollDataset(test_dataset, seq_len)

    labels = []
    binary_labels = []
    frame_labels = []
    clean_prs = np.zeros((seq_len, num_pitches * len(dataset)))
    deg_prs = np.zeros((seq_len, num_pitches * len(dataset)))
    for i, data in enumerate(dataset):
        labels.append(data["deg_label"])
        binary_labels.append(0 if data["deg_label"] == 0 else 1)
        frame_labels.extend(data["changed_frames"])
        deg_prs[:, i * num_pitches : (i + 1) * num_pitches] = data["deg_pr"]
        clean_prs[:, i * num_pitches : (i + 1) * num_pitches] = data["clean_pr"]
    labels = np.array(labels)
    binary_labels = np.array(binary_labels)
    frame_labels = np.array(frame_labels)

    # Task 1
    outputs = np.zeros((len(binary_labels), 2))
    outputs[:] = [deg_counts[0], np.sum(deg_counts[1:])]

    loss = torch.nn.CrossEntropyLoss()
    task1 = loss(
        torch.from_numpy(outputs).float(), torch.from_numpy(binary_labels).long()
    )
    print(f"Task 1 loss = {task1}")
    if deg_counts[0] < 0.5:
        print(f"Task 1 rev F-measure = 0.0")
    else:
        precision = np.sum(1 - binary_labels) / len(binary_labels)
        print(f"Task 1 rev F-measure = {(1 + precision) / (2 * precision)}")

    # Task 2
    outputs = np.zeros((len(labels), 9))
    outputs[:] = deg_counts

    task2 = loss(torch.from_numpy(outputs).float(), torch.from_numpy(labels).long())
    print(f"Task 2 loss = {task2}")
    print(f"Task 2 acc = {np.mean(1 - binary_labels)}")

    # Task 3
    outputs = np.zeros((len(frame_labels), 2))
    outputs[:] = frame_counts

    task3 = loss(
        torch.from_numpy(outputs).float(), torch.from_numpy(frame_labels).long()
    )
    print(f"Task 3 loss = {task3}")
    if frame_counts[0] > 0.5:
        print(f"Task 3 F-measure = 0.0")
    else:
        precision = np.sum(frame_labels) / len(frame_labels)
        print(f"Task 3 F-measure = {(1 + precision) / (2 * precision)}")

    # Task 4
    outputs = np.zeros(np.shape(clean_prs))
    outputs[deg_prs == 0] = pr_outputs[1, 0]
    outputs[deg_prs == 1] = pr_outputs[1, 1]

    loss = torch.nn.BCEWithLogitsLoss()
    task4 = loss(torch.from_numpy(outputs).float(), torch.from_numpy(clean_prs).float())
    helpfulness = (0.5 * np.sum(binary_labels) + np.sum(1 - binary_labels)) / len(
        binary_labels
    )
    print(f"Task 4 loss = {task4}")
    print(f"Task 4 Helpfulness = {helpfulness}")
