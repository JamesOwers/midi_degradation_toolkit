"""Utility functions and fields for dealing with note_dfs in mdtk format."""
import numpy as np
import pandas as pd

NOTE_DF_SORT_ORDER = ["onset", "track", "pitch", "dur", "velocity"]


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
        df = df.assign(track=0)  # Assign creates a copy so input is not changed

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
    df["offset"] = df["onset"] + df["dur"]
    offset = df["offset"].copy()

    for _, pitch_df in df.groupby(["track", "pitch"]):
        if len(pitch_df) < 2:
            continue

        # Each note's offset will go to the latest offset so far,
        # or be cut at the next note's onset
        cum_max = pitch_df["offset"].cummax()
        offset.loc[pitch_df.index] = cum_max.clip(
            upper=pitch_df["onset"].shift(-1, fill_value=cum_max.iloc[-1])
        )

    # Fix dur based on offsets and remove offset column
    df["dur"] = offset - df["onset"]
    df = df.loc[df["dur"] != 0, NOTE_DF_SORT_ORDER]
    df = df.reset_index(drop=True)

    return df


def get_random_excerpt(
    note_df,
    min_notes=10,
    excerpt_length=5000,
    first_onset_range=(0, 200),
    iterations=10,
):
    """
    Take a random excerpt from the given note_df, using np.random. The excerpt
    is created as follows:

    1. Pick a note at random from the input df, excluding the last `min_notes`
       notes.
    2. Take all notes which onset within `excerpt_length` ms of that note.
    3. If the excerpt does not contain at least `min_notes` notes, repeat steps
       1 and 2 until you have drawn `iterations` invalid excerpts. In that
       case, return None.
    4. If you have a valid excerpt, shift its notes so that the first note's
       onset is at time 0, then add a random number within `first_onset_range`
       to each onset.
    5. Return the resulting excerpt.

    Parameters
    ----------
    note_df : pd.DataFrame
        The input note_df, from which we want a random excerpt.

    min_notes : int
        The minimum number of notes that must be contained in a valid excerpt.

    excerpt_length : int
        The length of the resulting excerpt, in ms. All notes which onset
        within this amount of time after a randomly chosen note will be
        included in the returned excerpt.

    first_onset_range : tuple(int, int)
        The range from which to draw a random number to add to the first note's
        onset (in ms), rather than having the chosen excerpt begin at time 0.

    iterations : int
        How many times to try to obtain a valid excerpt before giving up and
        returning None.

    Returns
    -------
    excerpt : pd.DataFrame
        A random excerpt from the given note_df. None if no valid excerpt was
        found within `iterations` attempts.
    """
    if len(note_df) < min_notes or iterations == 0:
        return None

    for _ in range(iterations):
        note_index = np.random.choice(list(note_df.index.values)[:-min_notes])
        first_onset = note_df.loc[note_index]["onset"]
        excerpt = pd.DataFrame(
            note_df.loc[
                note_df["onset"].between(first_onset, first_onset + excerpt_length)
            ]
        )

        # Check for validity of excerpt
        if len(excerpt) < min_notes:
            excerpt = None
        else:
            break

    if excerpt is None:
        return None

    onset_shift = np.random.randint(first_onset_range[0], first_onset_range[1])
    excerpt["onset"] += onset_shift - first_onset
    excerpt = excerpt.reset_index(drop=True)
    return excerpt
