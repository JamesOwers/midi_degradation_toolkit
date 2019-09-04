"""Code to perform the degradations i.e. edits to the midi data"""
import sys
import warnings
import numpy as np
from numpy.random import randint, choice



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
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.

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



def split_range_sample(split_range, p=None):
    """
    Return a value sampled randomly from the given list of ranges. It is
    implemented to first sample a range from the list of ranges `split_range`,
    and then sample uniformly from the selected range. The sample of each range
    is proportional to the size of the range, by default, such that the
    resulting sample is akin to sampling uniformly from the range defined by
    taking the union of the ranges supplied in `split_range`.
    
    Parameters
    ----------
    split_range : list(tuple)
        A list of [min, max) tuples defining ranges from which to sample.
        
    p : list(float)
        If given, should be a list the same length as split_range, and
        contains the probability of sampling from each range. p will be
        normalized before use. Defaults to None.
        
    Returns
    -------
    samp : int
        An integer sampled from the given split range.
    """
    if p is not None:
        p = p / np.sum(p)
    else:
        range_sizes = [rr[1] - rr[0] for rr in split_range]
        total_range = sum(range_sizes)
        p = [range_size / total_range for range_size in range_sizes]
    index = choice(range(len(split_range)), p=p)
    samp = randint(split_range[index][0], split_range[index][1])
    return samp



@set_random_seed
def pitch_shift(excerpt, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH,
                distribution=None):
    """
    Shift the pitch of one note from the given excerpt.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.

    min_pitch : int
        The minimum pitch to which a note may be shifted.

    max_pitch : int
        The maximum pitch to which a note may be shifted.
        
    distribution : list(float)
        If given, a list describing the distribution of pitch shifts.
        Element (len(distribution) // 2) refers to the note's original
        pitch, and will be set to 0. Additionally, pitches outside of the
        range [min_pitch, max_pitch] will also be set to 0. The distribution
        will then be normalized to sum to 1, and used to generate a new
        pitch. Defaults to None, which implies a uniform distribution.

    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.


    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with the pitch of one note changed.
    """
    if excerpt.note_df.shape[0] == 0:
        warnings.warn('WARNING: No notes to pitch shift. Returning None.',
                      category=UserWarning)
        return None
    
    # Assume all notes can be shifted initially
    valid_notes = list(range(excerpt.note_df.shape[0]))
    
    # If distribution is being used, some notes may not be possible to pitch
    # shift. This is because the distribution supplied would only allow them
    # to be shifted outside of the supplied (min, max) pitch range. For example
    # A distribution [0, 0, 1] always shifts up one semitone; a note with
    # pitch equal to max_pitch can't be shifted with this distribution.
    if distribution is not None:
        assert all([dd >= 0 for dd in distribution]), ('A value in supplied '
                       'distribution is negative.')
        zero_idx = len(distribution) // 2
        distribution[zero_idx] = 0
        
        if np.sum(distribution) == 0:
            warnings.warn('WARNING: distribution contains only 0s after '
                          'setting distribution[zero_idx] value to 0. '
                          'Returning None.')
            return None
        
        nonzero_indices = np.nonzero(distribution)[0]
        
        lowest_idx = nonzero_indices[0]
        highest_idx = nonzero_indices[-1]
        
        min_pitch_shift = zero_idx - lowest_idx
        max_pitch_shift = highest_idx - zero_idx
        
        max_to_sample = max_pitch + min_pitch_shift
        min_to_sample = min_pitch - max_pitch_shift
        
        valid_notes = excerpt.note_df.index[excerpt.note_df['pitch']
                                            .between(min_to_sample,
                                                     max_to_sample)].tolist()
        
        if not valid_notes:
            warnings.warn('WARNING: No valid pitches to shift given '
                          f'min_pitch {min_pitch}, max_pitch {max_pitch}, '
                          f'and distribution {distribution} (after setting '
                          'distribution[zero_idx] to 0). Returning None.')
            return None
        
    degraded = excerpt.copy()

    # Sample a random note
    note_index = valid_notes[randint(len(valid_notes))]
    pitch = degraded.note_df.loc[note_index, 'pitch']
    
    # Shift its pitch
    if distribution is None:
        # Uniform distribution
        while degraded.note_df.loc[note_index, 'pitch'] == pitch:
            degraded.note_df.loc[note_index, 'pitch'] = randint(min_pitch,
                                                                max_pitch + 1)
    else:
        zero_idx = len(distribution) // 2
        pitches = np.array(range(pitch - zero_idx,
                                 pitch - zero_idx + len(distribution)))
        distribution[zero_idx] = 0
        distribution = np.where(pitches < min_pitch, 0, distribution)
        distribution = np.where(pitches > max_pitch, 0, distribution)
        distribution = distribution / np.sum(distribution)
        degraded.note_df.loc[note_index, 'pitch'] = choice(pitches,
                                                           p=distribution)

    return degraded



@set_random_seed
def time_shift(excerpt, min_shift=50, max_shift=np.inf):
    """
    Shift the onset and offset times of one note from the given excerpt,
    leaving its duration unchanged.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.

    min_shift : int
        The minimum amount by which the note will be shifted. Defaults to 50.
        
    max_shift : int
        The maximum amount by which the note will be shifted. Defaults to
        infinity.

    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.


    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with the timing of one note changed,
        or None if there are no notes that can be changed given the
        parameters.
    """
    onset = excerpt.note_df['onset']
    offset = onset + excerpt.note_df['dur']
    end_time = offset.max()
    
    # Shift earlier
    earliest_earlier_onset = (onset - (max_shift - 1)).clip(lower=0)
    latest_earlier_onset = ((onset - (min_shift - 1))
                                .clip(lower=earliest_earlier_onset,
                                      upper=onset))
    
    # Shift later
    latest_later_onset = onset + (((end_time + 1) - offset)
                                      .clip(upper=max_shift))
    earliest_later_onset = ((onset + min_shift)
                                .clip(lower=onset + 1,
                                      upper=latest_later_onset))
    
    # Find valid notes
    valid = ((earliest_earlier_onset < latest_earlier_onset) |
             (earliest_later_onset < latest_later_onset))
    valid_notes = list(valid.index[valid])
    
    if not valid_notes:
        warnings.warn('WARNING: No valid notes to time shift. Returning ' +
                      'None.', category=UserWarning)
        return None
    
    # Sample a random note
    index = choice(valid_notes)
    onset = split_range_sample([(earliest_earlier_onset[index],
                                 latest_earlier_onset[index]),
                                (earliest_later_onset[index],
                                 latest_later_onset[index])])
    
    degraded = excerpt.copy()
    
    degraded.note_df.loc[index, 'onset'] = onset
    
    return degraded



@set_random_seed
def onset_shift(excerpt, min_shift=50, max_shift=np.inf, min_duration=50,
                max_duration=np.inf):
    """
    Shift the onset time of one note from the given excerpt.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.
        
    min_shift : int
        The minimum amount by which the onset time will be changed. Defaults
        to 50.
        
    max_shift : int
        The maximum amount by which the onset time will be changed. Defaults
        to infinity.
        
    min_duration : int
        The minimum duration for the resulting note. Defaults to 50.
        
    max_duration : int
        The maximum duration for the resulting note. Defaults to infinity.
        (The offset time will never go beyond the current last offset
        in the excerpt.)

    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.

    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with the onset time of one note changed.
    """
    onset = excerpt.note_df['onset']
    offset = onset + excerpt.note_df['dur']
    
    # Lengthen bounds (decrease onset)
    earliest_lengthened_onset = ((offset - max_duration)
                                     .clip(lower=onset - max_shift)
                                     .clip(lower=0))
    latest_lengthened_onset = ((onset - max(min_shift - 1, 0))
                                   .clip(upper=offset - (min_duration - 1),
                                         lower=earliest_lengthened_onset))
    
    # Shorten bounds (increase onset)
    latest_shortened_onset = ((offset - (min_duration - 1))
                                  .clip(upper=onset + (max_shift + 1)))
    earliest_shortened_onset = ((onset + max(min_shift, 1))
                                    .clip(lower=offset - max_duration,
                                          upper=latest_shortened_onset))
    
    # Find valid notes
    valid = ((earliest_lengthened_onset < latest_lengthened_onset) |
             (earliest_shortened_onset < latest_shortened_onset))
    valid_notes = list(valid.index[valid])
            
    if not valid_notes:
        warnings.warn('WARNING: No valid notes to onset shift. Returning ' +
                      'None.', category=UserWarning)
        return None
    
    # Sample a random note
    index = choice(valid_notes)
    onset = split_range_sample([(earliest_lengthened_onset[index],
                                 latest_lengthened_onset[index]),
                                (earliest_shortened_onset[index],
                                 latest_shortened_onset[index])])
    
    degraded = excerpt.copy()
    
    degraded.note_df.loc[index, 'onset'] = onset
    degraded.note_df.loc[index, 'dur'] = offset[index] - onset
    
    return degraded



@set_random_seed
def offset_shift(excerpt, min_shift=50, max_shift=np.inf, min_duration=50,
                 max_duration=np.inf):
    """
    Shift the offset time of one note from the given excerpt.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.
        
    min_shift : int
        The minimum amount by which the offset time will be changed. Defaults
        to 50.
        
    max_shift : int
        The maximum amount by which the offset time will be changed. Defaults
        to infinity.
        
    min_duration : int
        The minimum duration for the resulting note. Defaults to 50.
        
    max_duration : int
        The maximum duration for the resulting note. Defaults to infinity.
        (The offset time will never go beyond the current last offset
        in the excerpt.)

    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.


    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with the offset time of one note changed.
    """
    onset = excerpt.note_df['onset']
    duration = excerpt.note_df['dur']
    end_time = (onset + duration).max()

    # Lengthen bounds (increase duration)
    shortest_lengthened_dur = ((duration + max(min_shift, 1))
                                   .clip(lower=min_duration))
    longest_lengthened_dur = ((duration + (max_shift + 1))
                                  .clip(upper=(end_time + 1) - onset)
                                  .clip(upper=max_duration + 1)
                                  .clip(lower=shortest_lengthened_dur))
    
    # Shorten bounds (decrease duration)
    shortest_shortened_dur = (duration - max_shift).clip(lower=min_duration)
    longest_shortened_dur = ((duration - (min_shift - 1))
                                 .clip(upper=duration)
                                 .clip(lower=shortest_shortened_dur,
                                       upper=max_duration + 1))
    
    # Find valid notes
    valid = ((shortest_lengthened_dur < longest_lengthened_dur) |
             (shortest_shortened_dur < longest_shortened_dur))
    valid_notes = list(valid.index[valid])
            
    if not valid_notes:
        warnings.warn('WARNING: No valid notes to offset shift. Returning ' +
                      'None.', category=UserWarning)
        return None
    
    # Sample a random note
    index = choice(valid_notes)
    
    duration = split_range_sample([(shortest_shortened_dur[index],
                                    longest_shortened_dur[index]),
                                   (shortest_lengthened_dur[index],
                                    longest_lengthened_dur[index])])
        
    degraded = excerpt.copy()
    
    degraded.note_df.loc[index, 'dur'] = duration
    
    return degraded



@set_random_seed
def remove_note(excerpt):
    """
    Remove one note from the given excerpt.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.

    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.


    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with one note removed.
    """
    if excerpt.note_df.shape[0] == 0:
        warnings.warn('WARNING: No notes to remove. Returning None.',
                      category=UserWarning)
        return None
        
    degraded = excerpt.copy()

    # Sample a random note
    note_index = choice(list(degraded.note_df.index))

    # Remove that note
    degraded.note_df.drop(note_index, inplace=True)
    degraded.note_df.reset_index(drop=True, inplace=True)

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
        The maximum pitch at which a note may be added.
    min_duration : int
        The minimum duration for the note to be added. Defaults to 50.
    max_duration : int
        The maximum duration for the added note. Defaults to infinity.
        (The offset time will never go beyond the current last offset
        in the excerpt.)
    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.


    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with one note added.
    """
    degraded = excerpt.copy()

    end_time = degraded.note_df[['onset', 'dur']].sum(axis=1).max()

    pitch = randint(min_pitch, max_pitch + 1)
    track = None

    if min_duration > end_time:
        onset = 0
        duration = min_duration
    elif degraded.note_df.shape[0] == 0:
        onset = 0
        duration = randint(min_duration, min(max_duration + 1, sys.maxsize))
    else:
        onset = randint(degraded.note_df['onset'].min(),
                        end_time - min_duration)
        duration = randint(min_duration,
                           min(end_time - onset, max_duration + 1))

    # Track is random one of existing tracks
    try:
        track = choice(degraded.note_df['track'].unique())
    except KeyError:  # No track col in df
        track = 0
    except ValueError:  # Empty dataframe
        track = 0

    degraded.note_df = degraded.note_df.append({'pitch': pitch,
                                                'onset': onset,
                                                'dur': duration,
                                                'track': track},
                                               ignore_index=True)
    return degraded



@set_random_seed
def split_note(excerpt, min_duration=50, num_splits=1):
    """
    Split one note from the excerpt into two or more notes of equal
    duration.
    
    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.
        
    min_duration : int
        The minimum length for any of the resulting notes.
        
    num_splits : int
        The number of splits to make in the chosen note. The note will
        be split into (num_splits+1) shorter notes.
        
    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.
        
    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with one note split.
    """
    if excerpt.note_df.shape[0] == 0:
        warnings.warn('WARNING: No notes to split. Returning None.',
                      category=UserWarning)
        return None
    
    # Find all splitable notes
    long_enough = excerpt.note_df['dur'] >= min_duration * (num_splits + 1)
    valid_notes = list(long_enough.index[long_enough])
    
    if not valid_notes:
        warnings.warn('WARNING: No valid notes to split. Returning ' +
                      'None.', category=UserWarning)
        return None
    
    note_index = choice(valid_notes)
    
    degraded = excerpt.copy()
    
    short_duration_float = (degraded.note_df.loc[note_index, 'dur'] /
                            (num_splits + 1))
    pitch = degraded.note_df.loc[note_index, 'pitch']
    track = degraded.note_df.loc[note_index, 'track']
    this_onset = degraded.note_df.loc[note_index, 'onset']
    next_onset = this_onset + short_duration_float
    
    # Shorten original note
    degraded.note_df.loc[note_index]['dur'] = int(round(short_duration_float))
    
    # Add next notes (taking care to round correctly)
    for i in range(num_splits):
        this_onset = next_onset
        next_onset += short_duration_float
        degraded.note_df = degraded.note_df.append(
            {
                'pitch': pitch,
                'onset': int(round(this_onset)),
                'dur': int(round(next_onset)) - int(round(this_onset)),
                'track': track
            },
            ignore_index=True
        )
    
    return degraded



@set_random_seed
def join_notes(excerpt, max_gap=50):
    """
    Combine two notes of the same pitch and track into one.
    
    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.
        
    max_gap : int
        The maximum gap length, in ms, for 2 notes to be able to be joined.
        (They must always share the same pitch and track).
        
    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.
        
    Returns
    -------
    degraded : Composition
        A copy of the given excerpt, with one note split.
    """
    if excerpt.note_df.shape[0] < 2:
        warnings.warn('WARNING: No notes to join. Returning None.',
                      category=UserWarning)
        return None
    
    valid_prev = []
    valid_next = []
    
    for _, track_df in excerpt.note_df.groupby('track'):
        for _, pitch_df in track_df.groupby('pitch'):
            if len(pitch_df) < 2:
                continue
                
            # Get note gaps
            onset = pitch_df['onset']
            offset = onset + pitch_df['dur']
            gap_after = onset.shift(-1) - offset
            gap_after.iloc[-1] = np.inf
            
            # Save index of note before and after gap
            valid = gap_after <= max_gap
            valid_prev = list(valid.index[valid])
            valid_next = list(valid.index[valid.shift(1)==True])
                    
    if not valid_prev:
        warnings.warn('WARNING: No valid notes to join. Returning ' +
                      'None.', category=UserWarning)
        return None
    
    index = randint(len(valid_prev))
    
    prev_i = valid_prev[index]
    next_i = valid_next[index]
    
    degraded = excerpt.copy()
    
    # Extend first note
    degraded.note_df.loc[prev_i]['dur'] = (degraded.note_df.loc[next_i]['onset'] +
                                           degraded.note_df.loc[next_i]['dur'] -
                                           degraded.note_df.loc[prev_i]['onset'])
    
    # Drop 2nd note
    degraded.note_df.drop(next_i, inplace=True)
    degraded.note_df.reset_index(drop=True, inplace=True)
    
    return degraded



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
