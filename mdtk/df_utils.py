"""Utility functions and fields for dealing with note_dfs in mdtk format.
"""
import itertools

import pandas as pd


NOTE_DF_SORT_ORDER = ['onset', 'track', 'pitch', 'dur']


def clean_df(df, flatten_tracks=False, remove_overlaps=False):
    """
    Clean a given note_df by (optionally) flattening the tracks of all notes
    to 0, (optionally) removing overlaps between notes, and sorting the notes
    by ascending onset, track, pitch, and dur, finally removing all additional
    columns.

    Parameters
    ----------
    df : pd.DataFrame
        A note_df, with columns onset, track, pitch, and dur (all ints).

    flatten_tracks : boolean
        True to set the track of every note to 0. This will happen before
        overlaps are removed.

    remove_overlaps : boolean
        True to remove overlaps from the resulting dataframe by passing the df
        to remove_overlaps. This will create a situation where, for every
        (track, pitch) pair, for any point in time which there is a
        sustained note present in the input, there will be a sustained note
        in the returned df. Likewise for any point with a note onset. If True,
        the resulting df will be sorted, even if sort is False.

    Returns
    -------
    df : pd.DataFrame
        A cleaned version of the given df, as described.
    """
    if flatten_tracks:
        df.loc[:, 'track'] = 0

    df = df.sort_values(by=NOTE_DF_SORT_ORDER).reset_index(drop=True)

    if remove_overlaps:
        df = remove_overlaps(df)

    return df.loc[:, NOTE_DF_SORT_ORDER]


def remove_overlaps(df):
    """
    Returns a version of the given df with all overlaps removed as:

    For every (track, pitch) pair, for any point in time which there is a
    sustained note present in the input, there will be a sustained note
    in the returned df. Likewise for any point with a note onset.

    This relies on the given df being sorted already.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with at least the columns onset, pitch, track, and dur,
        sorted in that order.

    Returns
    -------
    df : pd.DataFrame
        A non-overlapping version of the given df, as described above.
    """
    if len(df) < 2:
        return df

    # Copy since we're going to be editing in place
    df = df.copy()

    # We'll work with onsets here, and fix dur at the end
    df.loc[:, 'offset'] = df['onset'] + df['dur']

    for track, track_df in df.groupby('track'):
        if len(track_df) < 2:
            continue

        for pitch, pitch_df in track_df.groupby('pitch'):
            if len(pitch_df) < 2:
                continue

            # Last of any offset in the current set of overlapping notes
            current_offset = pitch_df.iloc[0]['offset']
            # We will need to change the previous offset in the case of an overlap
            prev_idx = pitch_df.index[0]

            for idx, note in itertools.islice(pitch_df.iterrows(), 1, None):
                if current_offset > note.onset:
                    # Overlap found. Cut previous note and extend offset
                    # Changes here are performed in the original df
                    df.loc[prev_idx, 'offset'] = note.onset
                    current_offset = max(current_offset, note.offset)
                    df.loc[idx, 'offset'] = current_offset
                else:
                    # No overlap. Update latest offset.
                    current_offset = note.offset
                # Always iterate, but no need to update current_offset here,
                # because it will definitely be < next_note.onset (because sorted).
                prev_idx = idx

    # Fix dur based on offsets and remove offset column
    df.loc[:, 'dur'] = df['offset'] - df['onset']
    df = df.loc[df['dur'] != 0, ['onset', 'track', 'pitch', 'dur']]
    df = df.reset_index(drop=True)

    return df
