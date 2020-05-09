#!/usr/bin/env python
"""Script to measure the errors from a transcription system in order to create
a degraded MIDI dataset with the given proportions of degradations."""
import argparse
import glob
import os
import pandas as pd
import pickle
import numpy as np
import warnings

import pretty_midi

from mdtk import degradations, midi, data_structures, formatters
from mdtk.degradations import (MIN_PITCH_DEFAULT, MAX_PITCH_DEFAULT,
                               DEGRADATIONS, MIN_SHIFT_DEFAULT)

FILE_TYPES = ['mid', 'pkl', 'csv']


def load_file(filename, pr_min_pitch=MIN_PITCH_DEFAULT,
              pr_max_pitch=MAX_PITCH_DEFAULT, pr_time_increment=40):
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
    ext = os.path.splitext(os.path.basename(filename))[1]

    if ext == '.mid':
        return midi.midi_to_df(filename)

    if ext == '.csv':
        return pd.read_csv(filename, names=['onset', 'track', 'pitch', 'dur'])

    if ext == '.pkl':
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
                             f"{pr_max_pitch}] = "
                             f"{pr_min_pitch - pr_max_pitch + 1}")
            
        piano_roll = np.vstack((note_pr, onset_pr))
        return formatters.double_pianoroll_to_df(
            piano_roll, min_pitch=pr_min_pitch, max_pitch=pr_max_pitch,
            time_increment=pr_time_increment)

    raise NotImplementedError(f'Extension {ext} not supported.')



def get_note_degs(gt_note, trans_note):
    """
    Get the count of each degradation given a ground truth note and a
    transcribed note.
    
    Parameters
    ----------
    gt_note : dict
        The ground truth note, with integer fields onset, pitch, track,
        and dur.

    trans_note : dict
        The corresponding transcribed note, with integer fields onset,
        pitch, track, and dur.

    Returns
    -------
    deg_counts : np.array(float)
        The count of each degradation between the notes, for the set
        of degradations which lead to the smallest total number of
        degradations. If multiple sets of degradations lead to the
        ground truth in the same total number of degradations, the mean
        of those counts is returned. Indices are in order of
        mdtk.degradations.DEGRADATIONS.
    """
    deg_counts = np.zeros(len(DEGRADATIONS))

    # Pitch shift
    if gt_note['pitch'] != trans_note['pitch']:
        deg_counts[list(DEGRADATIONS).index('pitch_shift')] = 1

    # Time shift
    if abs(gt_note['dur'] - trans_note['dur']) < MIN_SHIFT_DEFAULT:
        if abs(gt_note['onset'] - trans_note['onset']) < MIN_SHIFT_DEFAULT:
            return deg_counts
        deg_counts[list(DEGRADATIONS).index('time_shift')] = 1
        return deg_counts

    # Onset shift
    if abs(gt_note['onset'] - trans_note['onset']) >= MIN_SHIFT_DEFAULT:
        deg_counts[list(DEGRADATIONS).index('onset_shift')] = 1

    # Offset shift
    gt_offset = gt_note['onset'] + gt_note['dur']
    trans_offset = trans_note['onset'] + trans_note['dur']
    if abs(gt_offset - trans_offset) >= MIN_SHIFT_DEFAULT:
        deg_counts[list(DEGRADATIONS).index('offset_shift')] = 1

    return deg_counts



def get_excerpt_degs_recursive(gt_excerpt, trans_excerpt, known=dict()):
    """
    Get the count of each degradation given a ground truth excerpt and a
    transcribed excerpt.

    Parameters
    ----------
    gt_excerpt : pd.DataFrame
        The ground truth data frame.

    trans_excerpt : pd.DataFrame
        The corresponding transcribed dataframe.
        
    known : dict(tuple(tuple, tuple) -> np.array(int))
        For top-down dynamic programming, a tuple of the remaining gt
        row indices and the remaining transcription row indices, mapped
        to a tuple of precalculated deg_counts.

    Returns
    -------
    deg_counts : np.array(float)
        The count of each degradation in this transcription, for the set
        of degradations which lead to the smallest total number of
        degradations. If multiple sets of degradations lead to the
        ground truth in the same total number of degradations, the mean
        of those counts is returned. Indices are in order of
        mdtk.degradations.DEGRADATIONS.
    """
    # Base case 1: gt is empty
    if len(gt_excerpt) == 0:
        deg_counts = np.zeros(len(DEGRADATIONS))
        deg_counts[list(DEGRADATIONS).index('add_note')] += len(trans_excerpt)
        return deg_counts

    # Base case 2: transcription is empty
    if len(trans_excerpt) == 0:
        deg_counts = np.zeros(len(DEGRADATIONS))
        deg_counts[list(DEGRADATIONS).index('remove_note')] += len(gt_excerpt)
        return deg_counts

    # Dynamic programming short-circuit step
    key = (tuple(gt_excerpt.index.values),
           tuple(trans_excerpt.index.values))
    # This try except is faster than checking in and then returning
    try:
        return known[key]
    except:
        pass

    # Recursive step - for every pair of notes
    # TODO: idea:
    # First, precalculate all note-to-note diffs, then graph search
    # between them somehow
    min_count = np.inf
    num_min = 0
    deg_counts = np.zeros(len(DEGRADATIONS))
    for gt_idx, gt_note in gt_excerpt.iterrows():
        for trans_idx, trans_note in trans_excerpt.iterrows():
            gt_excerpt_new = gt_excerpt.drop(gt_idx)
            trans_excerpt_new = trans_excerpt.drop(trans_idx)

            # Caluculate degs
            note_key = (tuple([gt_idx]), tuple([trans_idx]))
            try:
                deg_counts_this = known(note_key)
            except:
                deg_counts_this = get_note_degs(gt_note, trans_note)
                known[note_key] = deg_counts_this
            deg_counts_this += get_excerpt_degs_recursive(
                gt_excerpt_new, trans_excerpt_new, known=known
            )

            # Update the minimum number of degs
            num_degs = np.sum(deg_counts_this)
            if num_degs < min_count:
                min_count = num_degs
                num_min = 1
                deg_counts = deg_counts_this
            elif num_degs == min_count:
                num_min += 1
                deg_counts += deg_counts_this
                
    # TODO: special checks for split and join

    # Average across each path to get to the min
    deg_counts /= num_min

    # Update known dict for dynamic programming
    known[key] = deg_counts
    
    return deg_counts



def get_excerpt_degs(gt_excerpt, trans_excerpt):
    """
    Get the count of each degradation given a ground truth excerpt and a
    transcribed excerpt.

    Parameters
    ----------
    gt_excerpt : pd.DataFrame
        The ground truth data frame.

    trans_excerpt : pd.DataFrame
        The corresponding transcribed dataframe.

    Returns
    -------
    degs : np.array(float)
        The count of each degradation in this transcription, in the order
        given by mdtk.degradations.DEGRADATIONS.

    clean : int
        1 if the sum of degs is 0. 0 Otherwise.
    """
    deg_counts = get_excerpt_degs_recursive(gt_excerpt, trans_excerpt)

    clean = 1 if np.sum(deg_counts) == 0 else 0
    return deg_counts, clean



def get_proportions(gt, trans, length=5000, min_notes=10):
    """
    Get the proportions of each degradation given a ground truth file and its
    transcription.
    
    Parameters
    ----------
    gt : string
        The filename of a ground truth musical score.
        
    trans : string
        The filename of a transciption of the given ground truth.
        
    length : int
        The length of the excerpts to grab in ms (plus sustains).
        
    min_notes : int
        The minimum number of notes required for an excerpt to be valid.
        
    Returns
    -------
    proportions : list(float)
        The rough proportion of excerpts from the ground truth with each
        degradation present in the transcription, in the order given by
        mdtk.degradations.DEGRADATIONS.
        
    clean : float
        The rough proportion of excerpts from the ground truth whose
        transcription is correct.
    """
    num_excerpts = 0
    deg_counts = np.zeros(len(DEGRADATIONS))
    clean_count = 0

    gt_df = load_file(gt)
    trans_df = load_file(trans)

    # Take each excerpt
    for idx, note in gt_df.iterrows():
        note_onset = note['onset']
        gt_excerpt = pd.DataFrame(
            gt_df.loc[gt_df['onset'].between(note_onset, note_onset + length)]
        )
        gt_excerpt['onset'] = gt_excerpt['onset'] - note_onset

        # Check for validity
        if len(gt_excerpt) < min_notes:
            continue

        # Here, we have a valid excerpt. Find its transcription.
        num_excerpts += 1
        trans_excerpt = pd.DataFrame(
            trans_df.loc[trans_df['onset'].between(note_onset,
                                                   note_onset + length)]
        )
        trans_excerpt['onset'] = trans_excerpt['onset'] - note_onset

        degs, clean = get_excerpt_degs(gt_excerpt, trans_excerpt)
        deg_counts += degs
        clean_count += clean

    # Divide number of errors by the number of possible excerpts
    proportions = deg_counts / num_excerpts
    clean = clean_count / num_excerpts
    return proportions, clean



def parse_args(args_input=None):
    parser = argparse.ArgumentParser(description="Measure errors from a "
                                     "transcription error in order to make "
                                     "a degraded MIDI dataset with the measure"
                                     " proportion of each degration.")
    
    parser.add_argument("--gt", help="The directory which contains the ground "
                        "truth musical scores or piano rolls.", required=True)
    parser.add_argument("--gt_ext", choices=FILE_TYPES, default=None,
                        help="Restrict the file type for the ground truths.")
    
    parser.add_argument("--trans", help="The directory which contains the "
                        "transcriptions.", required=True)
    parser.add_argument("--trans_ext", choices=FILE_TYPES, default=None,
                        help="Restrict the file type for the transcriptions.")
    
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
    
    # Get allowed file extensions
    trans_ext = [args.trans_ext] if args.trans_ext is not None else FILE_TYPES
    gt_ext = [args.gt_ext] if args.gt_ext is not None else FILE_TYPES
    
    trans = []
    for ext in trans_ext:
        trans.extend(glob.glob(os.path.join(args.trans, '*.' + ext)))
    
    proportion = np.zeros((len(DEGRADATIONS), 0))
    clean_prop = []
    
    for file in trans:
        basename = os.path.splitext(os.path.basename(file))[0]
        
        # Find gt file
        gt_list = []
        for ext in gt_ext:
            gt_list.extend(glob.glob(os.path.join(args.gt, basename + '.' + ext)))
            
        if len(gt_list) == 0:
            warnings.warn(f'No ground truth found for transcription {file}. Check'
                          ' that the file extension --gt_ext is correct (or not '
                          'given), and the dir --gt is correct. Searched for file'
                          f' {basename}.{gt_ext} in dir {args.gt}.')
            continue
        elif len(gt_list) > 1:
            warnings.warn(f'Multiple ground truths found for transcription {file}:'
                          f'{gt_list}. Defaulting to the first one. Try narrowing '
                          'down extensions with --gt_ext.')
        gt = gt_list[0]
        
        # TODO: Also get some parameters?
        prop, clean = get_proportions(gt, file, length=args.excerpt_length,
                                      min_notes=args.min_notes)
        proportion = np.vstack((proportions, prop))
        clean_prop.append(clean)
        
    proportion = np.mean(proportion, axis=0)
    clean = np.mean(clean_prop)
    
    # TODO: Write out to json file
    
    