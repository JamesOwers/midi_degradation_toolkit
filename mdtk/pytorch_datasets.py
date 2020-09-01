"""classes to use in conjunction with pytorch dataloaders"""
import logging

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from mdtk.degradations import MAX_PITCH_DEFAULT, MIN_PITCH_DEFAULT
from mdtk.formatters import FORMATTERS


def transform_to_torchtensor(output):
    return {key: torch.tensor(value) for key, value in output.items()}


# This is adapted from:
# https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
class CommandDataset(Dataset):
    def __init__(
        self,
        corpus_path,
        vocab,
        seq_len,
        encoding="utf-8",
        corpus_lines=None,
        in_memory=True,
        transform=None,
    ):
        """
        Returns command-based data for ACME tasks.

        Parameters
        ----------
        corpus_path : str
            Path to document containing the corpus of data. Each line is comma
            separated and contains the degraded command string, clean command
            string, then the degadation id label (0 is no degradation).

        vocab : Vocab class
            A Vocab class object (see CommandVocab in formatters.py). This is
            used to convert the string commands to integers and serves as an
            easy way of getting them back again.

        seq_len : int
            The maximum length for a sequence (all sequences will be padded
            to this length)

        encoding : str
            Encoding to use when opening the corpus file.

        corpus_lines : int
            Optional, if you know the number of lines in the corpus, supplying
            the number here saves counting the lines in the file.

        in_memory : bool
            Whether to store data in memory, or read from disk. N.B. If reading
            from disk, the __get_item__ method ignores the item index and just
            reads the next line of the file. This means batching will not be
            random when used with a dataloader.

        transform: func
            The output from __get_item__ is a dictionary of numpy arrays.
            The function transform is applied to the dictionary before it is
            returned so, for example, it can be used to convert all data to
            torch tensors.
        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.in_memory = in_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.transform = transform
        self.formatter = FORMATTERS["command"]

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not in_memory:
                for _ in tqdm.tqdm(f, desc="Counting nr corpus lines"):
                    self.corpus_lines += 1

            if in_memory:
                self.lines = [
                    line[:-1].split(",")
                    for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)
                ]
                self.corpus_lines = len(self.lines)

        if not in_memory:
            self.file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        deg_cmd, clean_cmd, deg_num = self.get_corpus_line(item)
        deg_cmd = self.tokenize_sentence(deg_cmd)
        deg_cmd = [self.vocab.sos_index] + deg_cmd + [self.vocab.eos_index]
        clean_cmd = self.tokenize_sentence(clean_cmd)
        clean_cmd = [self.vocab.sos_index] + clean_cmd + [self.vocab.eos_index]
        deg_num = int(deg_num)

        # Deg length and clipping
        deg_len = len(deg_cmd)
        if self.in_memory:
            item_nr_str = f" {item}"
        else:
            item_nr_str = " <unknown>"
        if deg_len > self.seq_len:
            logging.warning(
                f"Degraded command data point {item_nr_str} exceeds "
                f"given seq_len: {deg_len} > {self.seq_len}. "
                "Clipping."
            )
            deg_len = self.seq_len
            deg_cmd = deg_cmd[: self.seq_len]
        # Clean length and clipping
        clean_len = len(clean_cmd)
        if clean_len > self.seq_len:
            logging.warning(
                f"Clean command data point {item_nr_str} exceeds "
                f"given seq_len: {clean_len} > {self.seq_len}. "
                "Clipping."
            )
            clean_len = self.seq_len
            clean_cmd = clean_cmd[: self.seq_len]

        # Command padding
        deg_cmd += [self.vocab.pad_index for _ in range(self.seq_len - len(deg_cmd))]
        clean_cmd += [
            self.vocab.pad_index for _ in range(self.seq_len - len(clean_cmd))
        ]

        output = {
            self.formatter["deg_label"]: deg_cmd,
            self.formatter["clean_label"]: clean_cmd,
            self.formatter["task_labels"][0]: deg_num,
            "deg_len": deg_len,
            "clean_len": clean_len,
        }

        if self.transform is not None:
            output = self.transform(output)
        return output

    def tokenize_sentence(self, sentence):
        tokens = sentence.split()
        for ii, token in enumerate(tokens):
            tokens[ii] = self.vocab.stoi.get(token, self.vocab.unk_index)
        return tokens

    def get_corpus_line(self, item):
        if self.in_memory:
            deg_cmd, clean_cmd, deg_num = self.lines[item]
            return deg_cmd, clean_cmd, deg_num
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            deg_cmd, clean_cmd, deg_num = line[:-1].split(",")
            return deg_cmd, clean_cmd, deg_num


# This is adapted from
# https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
class PianorollDataset(Dataset):
    def __init__(
        self,
        corpus_path,
        seq_len,
        min_pitch=MIN_PITCH_DEFAULT,
        max_pitch=MAX_PITCH_DEFAULT,
        encoding="utf-8",
        corpus_lines=None,
        in_memory=True,
        transform=None,
    ):
        """
        Returns piano-roll-based data for ACME tasks.

        Parameters
        ----------
        corpus_path : str
            Path to document containing the corpus of data. Each line is comma
            separated and contains the degraded command string, clean command
            string, then the degadation id label (0 is no degradation).

        seq_len : int
            The maximum length for a piano-roll (all pianorolls will be 0-padded
            to this length)

        min_pitch : int
            The minimum pitch for a piano-roll.

        max_pitch : int
            The maximum pitch for a piano-roll.

        encoding : str
            Encoding to use when opening the corpus file.

        corpus_lines : int
            Optional, if you know the number of lines in the corpus, supplying
            the number here saves counting the lines in the file.

        in_memory : bool
            Whether to store data in memory, or read from disk. N.B. If reading
            from disk, the __get_item__ method ignores the item index and just
            reads the next line of the file. This means batching will not be
            random when used with a dataloader.

        transform: func
            The output from __get_item__ is a dictionary of numpy arrays.
            The function transform is applied to the dictionary before it is
            returned so, for example, it can be used to convert all data to
            torch tensors.
        """
        self.seq_len = seq_len
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

        self.in_memory = in_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.transform = transform
        self.formatter = FORMATTERS["pianoroll"]

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not in_memory:
                for _ in tqdm.tqdm(f, desc="Counting nr corpus lines"):
                    self.corpus_lines += 1

            if in_memory:
                self.lines = [
                    line[:-1].split(",")
                    for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)
                ]
                self.corpus_lines = len(self.lines)

        if not in_memory:
            self.file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        deg_pr, clean_pr, deg_num = self.get_corpus_line(item)
        deg_num = int(deg_num)
        deg_len, deg_pr = self.get_full_pr(deg_pr)
        clean_len, clean_pr = self.get_full_pr(clean_pr)
        changed_frames = np.array(
            [int(np.any(deg != clean)) for deg, clean in zip(deg_pr, clean_pr)]
        )

        output = {
            self.formatter["deg_label"]: deg_pr,
            self.formatter["clean_label"]: clean_pr,
            self.formatter["task_labels"][0]: deg_num,
            self.formatter["task_labels"][2]: changed_frames,
            "deg_len": deg_len,
            "clean_len": clean_len,
        }

        if self.transform is not None:
            output = self.transform(output)
        return output

    def get_full_pr(self, pr):
        note_pr = np.zeros((self.seq_len, 128))
        onset_pr = np.zeros((self.seq_len, 128))
        frames = pr.split("/")
        if len(frames) > self.seq_len:
            logging.warning(
                "Pianoroll data point exceeds given seq_len: "
                f"{len(frames)} > {self.seq_len}. Clipping."
            )
            frames = frames[: self.seq_len]
        for frame_num, frame in enumerate(frames):
            note_pitches, onset_pitches = frame.split("_")
            if note_pitches != "":
                note_pr[frame_num, list(map(int, note_pitches.split(" ")))] = 1
            if onset_pitches != "":
                onset_pr[frame_num, list(map(int, onset_pitches.split(" ")))] = 1
        return (
            len(frames),
            np.hstack(
                (
                    note_pr[:, self.min_pitch : self.max_pitch + 1],
                    onset_pr[:, self.min_pitch : self.max_pitch + 1],
                )
            ),
        )

    def get_corpus_line(self, item):
        if self.in_memory:
            deg_pr, clean_pr, deg_num = self.lines[item]
            return deg_pr, clean_pr, deg_num
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            deg_pr, clean_pr, deg_num = line[:-1].split(",")
            return deg_pr, clean_pr, deg_num
