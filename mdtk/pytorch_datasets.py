"""classes to use in conjunction with pytorch dataloaders"""
import pandas as pd
import numpy as np

#TODO: probably want to move everything but Pytorch dataset objects out of here


# TODO: later can auto detect vocab from corpus
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
    one_hot : np.ndarray
        An array containing the one-hot vectors describing the given DataFrame.
    """
    # Input validation
    assert max_time_shift % time_increment == 0, ("max_time_shift must be "
        "divisible by time_increment.")
    assert max_pitch >= min_pitch, "max_pitch must be >= min_pitch."
    assert time_increment > 0, "time_increment must be positive."
    assert max_time_shift > 0, "max_time_shift must be positive"

    note_off = df.loc[:, ['onset', 'pitch']]
    note_off['onset'] = note_off['onset'] + df['dur']
    note_off['cmd'] = note_off['pitch'].apply(lambda x: f'f{x}')
    note_off['cmd_type'] = 'f'
    note_on = df.loc[:, ['onset', 'pitch']]
    note_on['cmd'] = note_off['pitch'].apply(lambda x: f'o{x}')
    note_on['cmd_type'] = 'o'
    commands = pd.concat((note_on, note_off)).sort_values(
                   ['onset', 'cmd_type', 'pitch'],
                   ascending=[True, True, True])
    
    command_list = []
    current_onset = commands.onset.iloc[0]
    for idx, row in commands.iteritems():
        time_shift = row.onset - current_onset
        if time_shift != 0:
            command_list += [f't{time_shift}']
        command_list += [f'{row.cmd}']
        current_onset = row.onset

    return ' '.join(command_list)
