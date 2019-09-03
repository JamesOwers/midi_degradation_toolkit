#!/usr/bin/env python
"""Script to measure the errors from a transcription system in order to create
a degraded MIDI dataset with the given proportions of degradations."""
import argparse
import glob
import os
import pandas as pd
import pickle

import pretty_midi

from mdtk import degradations, midi, data_structures

FILE_TYPES = ['mid', 'pkl', 'csv']


def load_file(filename):
    """
    Load the given filename into a pandas dataframe.
    
    Parameters
    ----------
    filename : string
        The file to load into a dataframe.
        
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
        # TODO convert piano roll to df
        
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
    
    