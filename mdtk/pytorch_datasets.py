"""classes to use in conjunction with pytorch dataloaders"""
from torch.utils.data import Dataset
import tqdm
import torch
import random
import pandas as pd
import numpy as np
import os

#TODO: probably want to move everything but Pytorch dataset objects out of here

# Convenience function...
def diff_pd(df1, df2):
    """Identify differences between two pandas DataFrames"""
    assert (df1.columns == df2.columns).all(), \
        "DataFrame column names are different"
    if any(df1.dtypes != df2.dtypes):
        "Data Types are different, trying to convert"
        df2 = df2.astype(df1.dtypes)
    if df1.equals(df2):
        return None
    else:
        # need to account for np.nan != np.nan returning True
        diff_mask = (df1 != df2) & ~(df1.isnull() & df2.isnull())
        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]
        changed.index.names = ['id', 'col']
        difference_locations = np.where(diff_mask)
        changed_from = df1.values[difference_locations]
        changed_to = df2.values[difference_locations]
        return pd.DataFrame({'from': changed_from, 'to': changed_to},
                            index=changed.index)

# TODO: later can auto detect vocab from corpus if necessary
#       I'm doing things this way just for ability to change things
#       later with ease
class CommandVocab(object):
    def __init__(self, min_pitch=0,
                 max_pitch=127,
                 time_increment=40,
                 max_time_shift=4000, 
                 specials=["<pad>", "<unk>", "<eos>", "<sos>"]):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        # itos - integer to string
        self.itos = (list(specials) +  # special tokens
            [f'o{ii}' for ii in range(min_pitch, max_pitch+1)] +  # note_on
            [f'f{ii}' for ii in range(min_pitch, max_pitch+1)] +  # note_off
            [f't{ii}' for ii in range(time_increment, max_time_shift+1,
                                      time_increment)])  # time_shift
        self.stoi = {tok: ii for ii, tok in enumerate(self.itos)}
    

def create_corpus_csvs(acme_dir, name, prefix, df_converter_func):
    """
    From a given acme dataset, create formatted csv files to use with
    our provided pytorch Dataset classes.

    Parameters
    ----------
    acme_dir : string
        The directory containing the acme data.

    name : string
        The name to print in the loading message.

    prefix : string
        The string to prepend to "_corpus_path" and "_corpus_lin_nr" columns
        in the resulting metadata.csv file, as well as to use in the names
        of the resulting corpus-specific csv files like:
        {split}_{prefix}_corpus.csv

    df_converter_func : function
        The function to convert from a pandas DataFrame to a string in the
        desired format.
    """
    fh_dict = {
        split: open(
            os.path.join(acme_dir, f'{split}_{prefix}_corpus.csv'
        ), 'w') for split in ['train', 'valid', 'test']
    }
    line_counts = {
        split: 0 for split in ['train', 'valid', 'test']
    }
    meta_df = pd.read_csv(os.path.join(acme_dir, 'metadata.csv'))
    for idx, row in tqdm.tqdm(meta_df.iterrows(), total=meta_df.shape[0],
                         desc=f'Creating {name} corpus'):
        alt_df = pd.read_csv(os.path.join(acme_dir, row.altered_csv_path),
                             header=None,
                             names=['onset', 'track', 'pitch', 'dur'])
        alt_str = df_converter_func(alt_df)
        clean_df = pd.read_csv(os.path.join(acme_dir, row.clean_csv_path),
                               header=None,
                               names=['onset', 'track', 'pitch', 'dur'])
        clean_str = df_converter_func(clean_df)
        deg_num = row.degradation_id
        split = row.split
        fh = fh_dict[split]
        fh.write(f'{alt_str},{clean_str},{deg_num}\n')
        meta_df.loc[idx, f'{prefix}_corpus_path'] = fh.name
        meta_df.loc[idx, f'{prefix}_corpus_line_nr'] = line_counts[split]
        line_counts[split] += 1
    meta_df.to_csv(os.path.join(acme_dir, 'metadata.csv'))


def df_to_pianoroll_str(df, time_increment=40):
    """
    Convert a given pandas DataFrame into a packed piano-roll representation:
    Each string will look like:
    
    "notes1_onsets1/notes2_onsets2/..."
    
    where notes and onsets are space-separated strings of pitches. notes
    contains those pitches which are present at each frame, and onsets
    contains those pitches which have and onset at a given frame.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame which we will convert into the piano-roll.

    time_increment : int
        The length of a single frame, in milliseconds.
    """
    # Input validation
    assert time_increment > 0, "time_increment must be positive."

    quant_df = df.loc[:, ['pitch']]
    quant_df['onset'] = (df['onset'] / time_increment).round().astype(int)
    quant_df['offset'] = (((df['onset'] + df['dur']) / time_increment)
                          .round().astype(int).clip(lower=quant_df['onset'] + 1))

    # Create piano rolls
    length = quant_df['offset'].max()
    max_pitch = quant_df['pitch'].max() + 1
    note_pr = np.zeros((length, max_pitch))
    onset_pr = np.zeros((length, max_pitch))
    for _, note in quant_df.iterrows():
        onset_pr[note.onset, note.pitch] = 1
        note_pr[note.onset:note.offset, note.pitch] = 1

    # Pack into format
    strings = []
    for note_frame, onset_frame in zip(note_pr, onset_pr):
        strings.append(' '.join(map(str, np.where(note_frame == 1)[0])) +
                       '_' +
                       ' '.join(map(str, np.where(onset_frame == 1)[0])))

    return '/'.join(strings)


def df_to_command_str(df, min_pitch=0, max_pitch=127, time_increment=40,
                      max_time_shift=4000):
    """
    Convert a given pandas DataFrame into a sequence commands, note_on (o),
    note_off (f), and time_shift (t). Each command is followed by a number:
    o10 means note_on<midinote 10>, t60 means time_shift<60 ms>. It is assumed
    df has been sorted by onset.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame which we will convert into commands.

    min_pitch : int
        The minimum pitch at which notes will occur.

    max_pitch : int
        The maximum pitch at which notes will occur.

    time_increment : int
        The length of a single frame, in milliseconds.

    max_time_shift : int
        The maximum shift length, in milliseconds. Must be divisible by
        time_increment. 

    Returns
    -------
    command_string : str
        The string containing a space separated list of commands.
    """
    # Input validation
    assert max_time_shift % time_increment == 0, ("max_time_shift must be "
        "divisible by time_increment.")
    assert max_pitch >= min_pitch, "max_pitch must be >= min_pitch."
    assert time_increment > 0, "time_increment must be positive."
    assert max_time_shift > 0, "max_time_shift must be positive."

    # TODO: This rounding may result in notes of length 0.
    note_off = df.loc[:, ['onset', 'pitch']]
    note_off['onset'] = note_off['onset'] + df['dur']
    note_off['cmd'] = note_off['pitch'].apply(lambda x: f'f{x}')
    note_off['cmd_type'] = 'f'
    note_on = df.loc[:, ['onset', 'pitch']]
    note_on['cmd'] = note_off['pitch'].apply(lambda x: f'o{x}')
    note_on['cmd_type'] = 'o'
    commands = pd.concat((note_on, note_off))
    commands['onset'] = ((commands['onset'] / time_increment)
                         .round().astype(int) * time_increment)
    commands = commands.sort_values(['onset', 'cmd_type', 'pitch'],
                                    ascending=[True, True, True])

    command_list = []
    current_onset = commands.onset.iloc[0]
    for idx, row in commands.iterrows():
        while current_onset != row.onset:
            time_shift = min(row.onset - current_onset, max_time_shift)
            command_list += [f't{time_shift}']
            current_onset += time_shift
        command_list += [f'{row.cmd}']

    return ' '.join(command_list)


def command_str_to_df(cmd_str):
    """
    Convert a given string of commands back to a pandas DataFrame.

    Parameters
    ----------
    cmd_str : str
        The string containing a space separated list of commands.

    Returns
    -------
    df : pd.DataFrame
        The pandas DataFrame representing the note data
    """    
    commands = cmd_str.split()
    note_on_pitch = []
    note_on_time = []
    note_off_pitch = []
    note_off_time = []
    curr_time = 0
    for cmd_str in commands:
        cmd = cmd_str[0]
        value = int(cmd_str[1:])
        if cmd == 'o':
            note_on_pitch += [value]
            note_on_time += [curr_time]
        elif cmd == 'f':
            note_off_pitch += [value]
            note_off_time += [curr_time]
        elif cmd == 't':
            curr_time += value
        else:
            raise ValueError(f'Invalid command {cmd}')
    df = pd.DataFrame(columns=['onset', 'track', 'pitch', 'dur'], dtype=int)
    for ii, (pitch, onset) in enumerate(zip(note_on_pitch, note_on_time)):
        note_off_idx = note_off_pitch.index(pitch)  # gets first instance
        note_off_pitch.pop(note_off_idx)
        off = note_off_time.pop(note_off_idx)
        dur = off - onset
        track = 0
        df.loc[ii] = [onset, track, pitch, dur]
    
    return df
    


def transform_to_torchtensor(output):
    return {key: torch.tensor(value) for key, value in output.items()}



# This is adapted from https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
class CommandDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8",
                 corpus_lines=None, in_memory=True, transform=None):
        """
        Returns command-based data for ACME tasks.

        Parameters
        ----------
        corpus_path : str
            Path to document containing the corpus of data. Each line is comma
            separated and contains the degraded command string, clean command
            string, then the degadation id label (0 is no degradation).

        vocab : Vocab class
            A Vocab class object (see CommandVocab above). This is used to
            convert the string commands to integers and serves as an easy
            way of getting them back again.

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

        Returns
        -------
        df : pd.DataFrame
            The pandas DataFrame representing the note data
        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.in_memory = in_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.transform = transform

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not in_memory:
                for _ in tqdm.tqdm(f, desc="Counting nr corpus lines"):
                    self.corpus_lines += 1

            if in_memory:
                self.lines = [line[:-1].split(",")
                              for line in tqdm.tqdm(f, desc="Loading Dataset",
                                                    total=corpus_lines)]
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

        deg_cmd = deg_cmd[:self.seq_len]
        deg_cmd += [self.vocab.pad_index for _ in 
                    range(self.seq_len - len(deg_cmd))]
        clean_cmd = clean_cmd[:self.seq_len]
        clean_cmd += [self.vocab.pad_index for _ in 
                      range(self.seq_len - len(clean_cmd))]

        output = {"deg_commands": deg_cmd,
                  "clean_commands": clean_cmd,
                  "deg_label": deg_num}

        # TODO: implement transform as in https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
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



# This is adapted from https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/dataset/dataset.py
class PianorollDataset(Dataset):
    def __init__(self, corpus_path, max_len, min_pitch=0, max_pitch=127,
                 encoding="utf-8", corpus_lines=None, in_memory=True,
                 transform=None):
        """
        Returns piano-roll-based data for ACME tasks.

        Parameters
        ----------
        corpus_path : str
            Path to document containing the corpus of data. Each line is comma
            separated and contains the degraded command string, clean command
            string, then the degadation id label (0 is no degradation).

        max_len : int
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
        self.max_len = max_len
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

        self.in_memory = in_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.transform = transform

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not in_memory:
                for _ in tqdm.tqdm(f, desc="Counting nr corpus lines"):
                    self.corpus_lines += 1

            if in_memory:
                self.lines = [line[:-1].split(",")
                              for line in tqdm.tqdm(f, desc="Loading Dataset",
                                                    total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not in_memory:
            self.file = open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        deg_pr, clean_pr, deg_num = self.get_corpus_line(item)
        deg_pr = self.get_full_pr(deg_pr)
        clean_pr = self.get_full_pr(clean_pr)

        output = {"deg_pr": deg_pr,
                  "clean_pr": clean_pr,
                  "deg_label": deg_num}

        # TODO: implement transform as in https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        if self.transform is not None:
            output = self.transform(output)
        return output 

    def get_full_pr(self, pr):
        note_pr = np.zeros((self.max_len, self.max_pitch - self.min_pitch + 1))
        onset_pr = np.zeros((self.max_len, self.max_pitch - self.min_pitch + 1))
        for frame_num, frame in enumerate(pr.split('/')):
            note_pitches, onset_pitches = frame.split('_')
            if note_pitches != '':
                note_pr[frame_num, list(map(int, note_pitches.split(' ')))] = 1
            if onset_pitches != '':
                onset_pr[frame_num, list(map(int, onset_pitches.split(' ')))] = 1
        return np.hstack((note_pr, onset_pr))
                
                

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
