"""Utility functions and fields for dealing with note_dfs in mdtk format.
"""
import itertools

import pandas as pd


NOTE_DF_SORT_ORDER = ['onset', 'track', 'pitch', 'dur']


def clean_df(df, single_track=False, non_overlapping=False):
    """
    Clean a given note_df by (optionally) flattening the tracks of all notes
    to 0, (optionally) removing overlaps between notes, and sorting the notes
    by ascending onset, track, pitch, and dur, finally removing all additional
    columns.

    Parameters
    ----------
    df : pd.DataFrame
        A note_df, with columns onset, track, pitch, and dur (all ints).

    single_track : boolean
        True to set the track of every note to 0. This will happen before
        overlaps are removed.

    non_overlapping : boolean
        True to remove overlaps from the resulting dataframe by passing the df
        to remove_pitch_overlaps. This will create a situation where, for every
        (track, pitch) pair, for any point in time which there is a
        sustained note present in the input, there will be a sustained note
        in the returned df. Likewise for any point with a note onset.

    Returns
    -------
    df : pd.DataFrame
        A cleaned version of the given df, as described.
    """
    if single_track:
        df = df.assign(track=0) # Assign creates a copy so input is not changed

    if non_overlapping:
        df = remove_pitch_overlaps(df)
    else:
        # Remove_pitch_overlaps already sorts
        df = df.sort_values(by=NOTE_DF_SORT_ORDER).reset_index(drop=True)

    # Return with correct column ordering
    return df.loc[:, NOTE_DF_SORT_ORDER]


def remove_pitch_overlaps(df):
    """
    Returns a version of the given df with all same-pitch overlaps removed.

    For every (track, pitch) pair, for any point in time which there is a
    sustained note present in the input, there will be a sustained note
    in the returned df. Likewise for any point with a note onset.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with at least the columns onset, pitch, track, and
        dur.

    Returns
    -------
    df : pd.DataFrame
        A non-overlapping version of the given df as described above, sorted by
        onset, pitch, track, and then dur.
    """
    if len(df) < 2:
        return df

    df = df.sort_values(by=NOTE_DF_SORT_ORDER).reset_index(drop=True)

    # We'll work with offsets here, and fix dur at the end
    df['offset'] = df['onset'] + df['dur']
    offset = df['offset'].copy()

    for track, track_df in df.groupby('track'):
        if len(track_df) < 2:
            continue

        for pitch, pitch_df in track_df.groupby('pitch'):
            if len(pitch_df) < 2:
                continue

            # Each note's offset will go to the latest offset so far,
            # or be cut at the next note's onset
            cum_max = pitch_df['offset'].cummax()
            offset.loc[pitch_df.index] = cum_max.clip(
                upper=pitch_df['onset'].shift(-1, fill_value=cum_max.iloc[-1])
            )

    # Fix dur based on offsets and remove offset column
    df['dur'] = offset - df['onset']
    df = df.loc[df['dur'] != 0, ['onset', 'track', 'pitch', 'dur']]
    df = df.reset_index(drop=True)

    return df
