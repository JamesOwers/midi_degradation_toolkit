#!/usr/bin/env python
"""Script to measure the errors from a transcription system in order to create
a degraded MIDI dataset with the given proportions of degradations."""
import argparse
import glob
import os
import pandas as pd
import pickle

import pretty_midi

from mdtk import degradations, midi, data_structures, formatters
from mdtk.degradations import MIN_PITCH, MAX_PITCH

FILE_TYPES = ['mid', 'pkl', 'csv']


def load_file(filename, pr_min_pitch=MIN_PITCH, pr_max_pitch=MAX_PITCH,
              pr_time_increment=40):
    """
    Load the given filename into a pandas dataframe.

    Parameters
    ----------
    filename : string
        The file to load into a dataframe.

    pr_min_pitch : int
        The minimum pitch for any piano roll, inclusive.

    pr_max_pitch : int
        The maximum pitch for any piano roll, inclusive.

    pr_time_increment : int
        The length of each frame of any piano roll.

    Return
    ------
    df : pandas dataframe
        A pandas dataframe representing the music from the given file.
    """
    ext = os.path.splitext(os.path.basename(file))[1]

    if ext == 'mid':
        return midi.mid_to_df(filename)

    if ext == 'csv':
        return data_structures.read_note_csv(filename)

    if ext == 'pkl':
        with open(filename, 'rb') as file:
            pkl = pickle.load(file)

        piano_roll = pkl['piano_roll']

        if piano_roll.shape[1] == (pr_min_pitch - pr_max_pitch + 1):
            # Normal piano roll only -- no onsets
            note_pr = piano_roll.astype(int)
            onset_pr = ((np.roll(note_pr, 1, axis=0) - note_pr) == -1)
            onset_pr[0] = note_pr[0]
            onset_pr = onset_pr.astype(int)

        elif piano_roll.shape[1] == 2 * (pr_min_pitch - pr_max_pitch + 1):
            # Piano roll with onsets
            note_pr = piano_roll[:, :piano_roll.shape[1] / 2].astype(int)
            onset_pr = piano_roll[:, piano_roll.shape[1] / 2:].astype(int)

        else:
            raise ValueError("Piano roll dimension 2 size ("
                             f"{piano_roll.shape[1]}) must be equal to 1 or 2"
                             f" times the given pitch range [{pr_min_pitch} - "
                             f"{pr_max_pitch}]")
            
        piano_roll = np.vstack((note_pr, onset_pr))
        return formatters.double_pianoroll_to_df(
            piano_roll, min_pitch=pr_min_pitch, max_pitch=pr_max_pitch,
            time_increment=pr_time_increment)

    raise NotImplementedError(f'Extension {ext} not supported.')



def get_proportions(gt, trans):
    """
    Get the proportions of each degradation given a ground truth file and its
    transcription.
    
    Parameters
    ----------
    gt : string
        The filename of a ground truth musical score.
        
    trans : string
        The filename of a transciption of the given ground truth.
        
    Returns
    -------
    proportions : list(float)
        The rough proportion of each degradation present in the transcription,
        in the order given by mdtk.degradations.get_degradations().
    """
    proportions = np.zeros(len(degradations.get_degradations()))
    
    gt_df = load_file(gt)
    trans_df = load_file(trans)
    
    # Match notes
    
    
    # Measure error for each matched note
    
    
    # Divide number of errors by length of piece
    
    pass



def parse_args(args_input=None):
    parser = argparse.ArgumentParser(description="Measure errors from a "
                                     "transcription error in order to make "
                                     "a degraded MIDI dataset with the measure"
                                     " proportion of each degration.")
    
    parser.add_argument("--gt", help="The directory which contains the ground "
                        "truth musical scores or piano rolls.")
    parser.add_argument("--gt_ext", choices=FILE_TYPES, default='mid',
                        help="The file type for the ground truth files.")
    
    parser.add_argument("--trans", help="The directory which contains the "
                        "transcriptions.")
    parser.add_argument("--trans_ext", choices=FILE_TYPES, default='mid',
                        help="The file type for the transcriptions.")
    
    # Pianoroll specific args
    parser.add_argument("--pr-min-pitch", type=int, default=21,
                        help="Minimum pianoroll pitch.")
    parser.add_argument("--pr-max-pitch", type=int, default=108,
                        help="Maximum pianoroll pitch.")
    
    # Excerpt arguments
    parser.add_argument('--excerpt-length', metavar='ms', type=int,
                        help='The length of the excerpt (in ms) to take from '
                        'each piece. The excerpt will start on a note onset '
                        'and include all notes whose onset lies within this '
                        'number of ms after the first note.', default=5000)
    parser.add_argument('--min-notes', metavar='N', type=int, default=10,
                        help='The minimum number of notes required for an '
                        'excerpt to be valid.')
    args = parser.parse_args(args=args_input)
    return args



if __name__ == '__main__':
    args = parse_args()
    
    trans = glob.glob(os.path.join(args.trans, '*.' + args.trans_ext))
    
    proportion = np.zeros((len(degradations.get_degradations()), 0))
    
    for file in trans:
        basename = os.path.splitext(os.path.basename(file))[0]
        gt = os.path.join(args.gt, basename + '.' + args.gt_ext)
        
        # TODO: Also get some parameters?
        proportion = np.vstack((proportions, get_proportions(gt, trans)))
        
    proportion = np.mean(proportion, axis=0)
    
    # TODO: Write out to json file
    
    