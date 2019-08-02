"""Code to perform the degradations i.e. edits to the midi data"""
import numpy as np
from numpy.random import randint, uniform, choice

from mdtk.data_structures import Composition



MIN_PITCH = 0
MAX_PITCH = 127



def set_random_seed(func, seed=None):
    """This is a function decorator which just adds the keyword argument `seed`
    to the end of the supplied function that it decorates. It seeds numpy's
    random state with the provided value before the call of the function.

    Parameters
    ----------
    func : function
        function to be decorated
    seed : int or None
        integer value to use for seed, if None leaves the random state as is

    Returns
    -------
    seeded_func : function
        The originally supplied function, but now with an aditional optional
        seed keyword argument.
    """
    def seeded_func(*args, seed=seed, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        return func(*args, **kwargs)
    return seeded_func



@set_random_seed
def pitch_shift(excerpt, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH):
    """
    Shift the pitch of one note from the given excerpt.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.
    min_pitch : int
        The minimum pitch to which a note may be shifted.
    max_pitch : int
        One greater than the maximum pitch to which a note may be shifted.
    seed : int
        A seed to be supplied to np.random.seed(), defaults to None

    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with the pitch of one note changed.
    """
    degraded = excerpt.copy()

    # Sample a random note
    note_index = randint(0, degraded.note_df.shape[0])

    # Shift its pitch (to something new)
    # TODO: I reckon this may update in place so the while loop will go forever
    original_pitch = degraded.note_df.loc[note_index, 'pitch']
    while degraded.note_df.loc[note_index, 'pitch'] == original_pitch:
        degraded.note_df.loc[note_index, 'pitch'] = randint(min_pitch,
                                                            max_pitch)

    return degraded



@set_random_seed
def time_shift(excerpt, params):
    """
    Shift the onset and offset times of one note from the given excerpt,
    leaving its duration unchanged.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.

    params : dict
        A dictionary containing parameters for the time shift. All used
        parameters keys will begin with 'time_shift_'. They include:
            None

    seed : int
        A seed to be supplied to np.random.seed(), defaults to None

    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with the timing of one note changed.
    """
    raise NotImplementedError()



@set_random_seed
def onset_shift(excerpt):
    """
    Shift the onset time of one note from the given excerpt.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.

    seed : int
        A seed to be supplied to np.random.seed(), defaults to None

    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with the onset time of one note changed.
    """
    raise NotImplementedError()



@set_random_seed
def offset_shift(excerpt):
    """
    Shift the offset time of one note from the given excerpt.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.

    seed : int
        A seed to be supplied to np.random.seed(), defaults to None


    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with the offset time of one note changed.
    """
    raise NotImplementedError()



@set_random_seed
def remove_note(excerpt):
    """
    Remove one note from the given excerpt.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.

    seed : int
        A seed to be supplied to np.random.seed(), defaults to None


    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with one note removed.
    """
    degraded = excerpt.copy()

    # Sample a random note
    note_index = randint(0, degraded.note_df.shape[0])

    # Remove that note
    (degraded.note_df
         .drop(note_index, inplace=True)
         .reset_index(drop=True, inplace=True)
    )

    return degraded



@set_random_seed
def add_note(excerpt, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH,
             min_duration=50, max_duration=np.inf):
    """
    Add one note to the given excerpt.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.
    min_pitch : int
        The minimum pitch at which a note may be added.
    max_pitch : int
        One greater than the maximum pitch at which a note may be added.
    min_duration : float
        The minimum duration for the note to be added. Defaults to 50.
    max_duration : float
        The maximum duration for the added note. Defaults to infinity.
        (The offset time will never go beyond the current last offset
        in the excerpt.)
    seed : int
        A seed to be supplied to np.random.seed(), defaults to None


    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with one note added.
    """
    degraded = excerpt.copy()

    end_time = degraded.note_df[['onset', 'dur']].sum(axis=1).max()

    pitch = randint(min_pitch, max_pitch)

    onset = uniform(degraded.note_df['onset'].min(), end_time - min_duration)
    duration = uniform(onset + min_duration,
                       max(end_time - onset, max_duration))

    # Track is random one of existing tracks
    track = choice(degraded.note_df['track'].unique())

    degraded.note_df = degraded.note_df.append({'pitch': pitch,
                                                'onset': onset,
                                                'dur': duration,
                                                'track': track},
                                               ignore_index=True)
    return degraded



@set_random_seed
def split_note(excerpt, params):
    """Split a note into two notes."""
    raise NotImplementedError()



@set_random_seed
def join_notes(excerpt, params):
    """Combine two notes into one note."""
    raise NotImplementedError()



DEGRADATIONS = {
    'pitch_shift': pitch_shift,
    'time_shift': time_shift,
    'onset_shift': onset_shift,
    'offset_shift': offset_shift,
    'remove_note': remove_note,
    'add_note': add_note,
    'split_note': split_note,
    'join_notes': join_notes
}



def get_degradations(degradation_list):
    """
    A convenience function to get degradation functions from strings.

    Parameters
    ----------
    degradation_list : list of strings
        list containing the names of the degradation functions to get

    Returns
    -------
    degradations : dict(string -> func)
        A dict containing the name of each degradation mapped to a pointer to
        the method which performs that degradation.
    """
    deg_funcs = [DEGRADATIONS[deg_str] for deg_str in degradation_list]
    return deg_funcs
