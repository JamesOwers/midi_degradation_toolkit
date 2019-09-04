"""Code to perform the degradations i.e. edits to the midi data"""
import sys
import warnings
import numpy as np
import pandas as pd
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
                distribution=None, inplace=False):
    """
    Shift the pitch of one note from the given excerpt.

    Parameters
    ----------
    excerpt : pd.DataFrame
        An excerpt from a piece of music.

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
        
    inplace : boolean
        True to edit the given DataFrame in place. False to create a copy.
        The result is returned either way.

    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.


    Returns
    -------
    degraded : pd.DataFrame
        A degradation of the excerpt, with the pitch of one note changed,
        or None if inplace=True or no notes can be pitch shifted.
    """
    if len(excerpt) == 0:
        warnings.warn('WARNING: No notes to pitch shift. Returning None.',
                      category=UserWarning)
        return None
    
    # Assume all notes can be shifted initially
    valid_notes = list(excerpt.index)
    
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
        
        valid_notes = excerpt.index[excerpt['pitch']
                                    .between(min_to_sample,
                                             max_to_sample)].tolist()
        
        if not valid_notes:
            warnings.warn('WARNING: No valid pitches to shift given '
                          f'min_pitch {min_pitch}, max_pitch {max_pitch}, '
                          f'and distribution {distribution} (after setting '
                          'distribution[zero_idx] to 0). Returning None.')
            return None
        
    if inplace:
        degraded = excerpt
    else:
        degraded = excerpt.copy()

    # Sample a random note
    note_index = valid_notes[randint(len(valid_notes))]
    pitch = degraded.loc[note_index, 'pitch']
    
    # Shift its pitch
    if distribution is None:
        # Uniform distribution
        while degraded.loc[note_index, 'pitch'] == pitch:
            degraded.loc[note_index, 'pitch'] = randint(min_pitch,
                                                        max_pitch + 1)
    else:
        zero_idx = len(distribution) // 2
        pitches = np.array(range(pitch - zero_idx,
                                 pitch - zero_idx + len(distribution)))
        distribution[zero_idx] = 0
        distribution = np.where(pitches < min_pitch, 0, distribution)
        distribution = np.where(pitches > max_pitch, 0, distribution)
        distribution = distribution / np.sum(distribution)
        degraded.loc[note_index, 'pitch'] = choice(pitches, p=distribution)

    if not inplace:
        return degraded



@set_random_seed
def time_shift(excerpt, min_shift=50, max_shift=np.inf, inplace=False):
    """
    Shift the onset and offset times of one note from the given excerpt,
    leaving its duration unchanged.

    Parameters
    ----------
    excerpt : pd.DataFrame
        A Composition object of an excerpt from a piece of music.

    min_shift : int
        The minimum amount by which the note will be shifted. Defaults to 50.
        
    max_shift : int
        The maximum amount by which the note will be shifted. Defaults to
        infinity.
        
    inplace : boolean
        True to edit the given DataFrame in place. False to create a copy.
        The result is returned either way.

    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.


    Returns
    -------
    degraded : pd.DataFrame
        A degradation of the excerpt, with the timing of one note changed,
        or None if there are no notes that can be changed given the
        parameters, or if inplace=True.
    """
    min_shift = max(min_shift, 1)
    
    onset = excerpt['onset']
    offset = onset + excerpt['dur']
    end_time = offset.max()
    
    # Shift earlier
    earliest_earlier_onset = (onset - (max_shift - 1)).clip(lower=0)
    latest_earlier_onset = onset - (min_shift - 1)
    
    # Shift later
    latest_later_onset = onset + (((end_time + 1) - offset)
                                      .clip(upper=max_shift))
    earliest_later_onset = onset + min_shift
    
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
    
    eeo = earliest_earlier_onset[index]
    leo = max(latest_earlier_onset[index], eeo)
    elo = earliest_later_onset[index]
    llo = max(latest_later_onset[index], elo)
    
    onset = split_range_sample([(eeo, leo), (elo, llo)])
    
    if inplace:
        degraded = excerpt
    else:
        degraded = excerpt.copy()
    
    degraded.loc[index, 'onset'] = onset
    
    if not inplace:
        return degraded



@set_random_seed
def onset_shift(excerpt, min_shift=50, max_shift=np.inf, min_duration=50,
                max_duration=np.inf, inplace=False):
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
        
    inplace : boolean
        True to edit the given DataFrame in place. False to create a copy.
        The result is returned either way.

    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.

    Returns
    -------
    degraded : Composition
        A degradation of the excerpt, with the onset time of one note changed.
    """
    min_shift = max(min_shift, 1)
    min_duration -= 1
    
    onset = excerpt['onset']
    offset = onset + excerpt['dur']
    
    # Lengthen bounds (decrease onset)
    earliest_lengthened_onset = ((offset - max_duration)
                                     .clip(lower=onset - max_shift)
                                     .clip(lower=0))
    latest_lengthened_onset = ((onset - (min_shift - 1))
                                   .clip(upper=offset - min_duration))
    
    # Shorten bounds (increase onset)
    latest_shortened_onset = ((offset - min_duration)
                                  .clip(upper=onset + (max_shift + 1)))
    earliest_shortened_onset = ((onset + min_shift)
                                    .clip(lower=offset - max_duration))
    
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
    
    elo = earliest_lengthened_onset[index]
    llo = max(latest_lengthened_onset[index], elo)
    eso = earliest_shortened_onset[index]
    lso = max(latest_shortened_onset[index], eso)
    
    onset = split_range_sample([(elo, llo), (eso, lso)])
    
    if inplace:
        degraded = excerpt
    else:
        degraded = excerpt.copy()
    
    degraded.loc[index, 'onset'] = onset
    degraded.loc[index, 'dur'] = offset[index] - onset
    
    if not inplace:
        return degraded



@set_random_seed
def offset_shift(excerpt, min_shift=50, max_shift=np.inf, min_duration=50,
                 max_duration=np.inf, inplace=False):
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
        
    inplace : boolean
        True to edit the given DataFrame in place. False to create a copy.
        The result is returned either way.

    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.


    Returns
    -------
    degraded : Composition
        A degradation of the excerpt, with the offset time of one note changed.
    """
    min_shift = max(min_shift, 1)
    max_duration += 1
    
    onset = excerpt['onset']
    duration = excerpt['dur']
    end_time = (onset + duration).max()

    # Lengthen bounds (increase duration)
    shortest_lengthened_dur = (duration + min_shift).clip(lower=min_duration)
    longest_lengthened_dur = ((duration + (max_shift + 1))
                                  .clip(upper=(end_time + 1) - onset)
                                  .clip(upper=max_duration))
    
    # Shorten bounds (decrease duration)
    shortest_shortened_dur = (duration - max_shift).clip(lower=min_duration)
    longest_shortened_dur = ((duration - (min_shift - 1))
                                 .clip(upper=max_duration))
    
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
    
    ssd = shortest_shortened_dur[index]
    lsd = max(longest_shortened_dur[index], ssd)
    sld = shortest_lengthened_dur[index]
    lld = max(longest_lengthened_dur[index], sld)
    
    duration = split_range_sample([(ssd, lsd), (sld, lld)])
        
    if inplace:
        degraded = excerpt
    else:
        degraded = excerpt.copy()
    
    degraded.loc[index, 'dur'] = duration
    
    if not inplace:
        return degraded



@set_random_seed
def remove_note(excerpt, inplace=False):
    """
    Remove one note from the given excerpt.

    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.
        
    inplace : boolean
        True to edit the given DataFrame in place. False to create a copy.
        The result is returned either way.

    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.


    Returns
    -------
    degraded : Composition
        A degradation of the excerpt, with one note removed.
    """
    if excerpt.shape[0] == 0:
        warnings.warn('WARNING: No notes to remove. Returning None.',
                      category=UserWarning)
        return None
        
    if inplace:
        degraded = excerpt
    else:
        degraded = excerpt.copy()

    # Sample a random note
    note_index = choice(list(degraded.index))

    # Remove that note
    degraded.drop(note_index, inplace=True)
    degraded.reset_index(drop=True, inplace=True)

    if not inplace:
        return degraded



@set_random_seed
def add_note(excerpt, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH,
             min_duration=50, max_duration=np.inf, inplace=False):
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
    inplace : boolean
        True to edit the given DataFrame in place. False to create a copy.
        The result is returned either way.
    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.


    Returns
    -------
    degraded : Composition
        A degradation of the excerpt, with one note added.
    """
    end_time = excerpt[['onset', 'dur']].sum(axis=1).max()

    pitch = randint(min_pitch, max_pitch + 1)
    track = None

    if min_duration > end_time:
        onset = 0
        duration = min_duration
    elif excerpt.shape[0] == 0:
        onset = 0
        duration = randint(min_duration, min(max_duration + 1, sys.maxsize))
    else:
        onset = randint(excerpt['onset'].min(),
                        end_time - min_duration)
        duration = randint(min_duration,
                           min(end_time - onset, max_duration + 1))

    # Track is random one of existing tracks
    try:
        track = choice(excerpt['track'].unique())
    except KeyError:  # No track col in df
        track = 0
    except ValueError:  # Empty dataframe
        track = 0

    if inplace:
        degraded = excerpt
        if len(degraded) > 0:
            index = max(degraded.index) + 1
        else:
            index = 0
        degraded.loc[index] = {'pitch': pitch,
                                       'onset': onset,
                                       'dur': duration,
                                       'track': track}
    else:
        degraded = excerpt.copy()
        degraded = degraded.append({'pitch': pitch,
                                                    'onset': onset,
                                                    'dur': duration,
                                                    'track': track},
                                                   ignore_index=True)
        
    if not inplace:
        return degraded



@set_random_seed
def split_note(excerpt, min_duration=50, num_splits=1, inplace=False):
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
        
    inplace : boolean
        True to edit the given DataFrame in place. False to create a copy.
        The result is returned either way.
        
    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.
        
    Returns
    -------
    degraded : Composition
        A degradation of the excerpt, with one note split.
    """
    if excerpt.shape[0] == 0:
        warnings.warn('WARNING: No notes to split. Returning None.',
                      category=UserWarning)
        return None
    
    # Find all splitable notes
    long_enough = excerpt['dur'] >= min_duration * (num_splits + 1)
    valid_notes = list(long_enough.index[long_enough])
    
    if not valid_notes:
        warnings.warn('WARNING: No valid notes to split. Returning ' +
                      'None.', category=UserWarning)
        return None
    
    note_index = choice(valid_notes)
    
    short_duration_float = (excerpt.loc[note_index, 'dur'] /
                            (num_splits + 1))
    pitch = excerpt.loc[note_index, 'pitch']
    track = excerpt.loc[note_index, 'track']
    this_onset = excerpt.loc[note_index, 'onset']
    next_onset = this_onset + short_duration_float
    
    # Add next notes (taking care to round correctly)
    pitches = [pitch] * num_splits
    onsets = [0] * num_splits
    durs = [0] * num_splits
    tracks = [track] * num_splits
    for i in range(num_splits):
        this_onset = next_onset
        next_onset += short_duration_float
        
        onsets[i] = int(round(this_onset))
        durs[i] = int(round(next_onset)) - int(round(this_onset))
        
    if inplace:
        degraded = excerpt
        if len(degraded) > 0:
            start = max(degraded.index) + 1
        else:
            start = 0
        
        for note_idx, df_idx in enumerate(range(start, start + num_splits)):
            degraded.loc[df_idx] = {'onset': onsets[note_idx],
                                            'track': tracks[note_idx],
                                            'pitch': pitches[note_idx],
                                            'dur': durs[note_idx]}
    else:
        degraded = excerpt.copy()
        new_df = pd.DataFrame({'onset': onsets,
                               'track': tracks,
                               'pitch': pitches,
                               'dur': durs})
        degraded = degraded.append(new_df, ignore_index = True)
        
    # Shorten original note
    degraded.loc[note_index]['dur'] = int(round(short_duration_float))
    
    if not inplace:
        return degraded



@set_random_seed
def join_notes(excerpt, max_gap=50, inplace=False):
    """
    Combine two notes of the same pitch and track into one.
    
    Parameters
    ----------
    excerpt : Composition
        A Composition object of an excerpt from a piece of music.
        
    max_gap : int
        The maximum gap length, in ms, for 2 notes to be able to be joined.
        (They must always share the same pitch and track).
        
    inplace : boolean
        True to edit the given DataFrame in place. False to create a copy.
        The result is returned either way.
        
    seed : int
        A seed to be supplied to np.random.seed(). Defaults to None, which
        leaves numpy's random state unchanged.
        
    Returns
    -------
    degraded : Composition
        A degradation of the excerpt, with one note split.
    """
    if excerpt.shape[0] < 2:
        warnings.warn('WARNING: No notes to join. Returning None.',
                      category=UserWarning)
        return None
    
    valid_prev = []
    valid_next = []
    
    for _, track_df in excerpt.groupby('track'):
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
    
    if inplace:
        degraded = excerpt
    else:
        degraded = excerpt.copy()
    
    # Extend first note
    degraded.loc[prev_i]['dur'] = (degraded.loc[next_i]['onset'] +
                                           degraded.loc[next_i]['dur'] -
                                           degraded.loc[prev_i]['onset'])
    
    # Drop 2nd note
    degraded.drop(next_i, inplace=True)
    degraded.reset_index(drop=True, inplace=True)
    
    if not inplace:
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
