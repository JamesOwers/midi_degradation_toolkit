"""Code to perform the degradations i.e. edits to the midi data"""
import sys
import warnings
import numpy as np
import pandas as pd
from numpy.random import randint, choice

from mdtk.data_structures import NOTE_DF_SORT_ORDER


MIN_PITCH_DEFAULT = 21
MAX_PITCH_DEFAULT = 108

MIN_SHIFT_DEFAULT = 100
MAX_SHIFT_DEFAULT = np.inf

MIN_DURATION_DEFAULT = 50
MAX_DURATION_DEFAULT = np.inf

TRIES_DEFAULT = 10

TRIES_WARN_MSG = ("WARNING: Generated invalid (overlapping) degraded excerpt "
                  "too many times. Try raising tries parameter (default 10). "
                  "Returning None.")


def set_random_seed(func, seed=None):
    """This is a function decorator which just adds the keyword argument `seed`
    to the end of the supplied function that it decorates. It seeds numpy's
    random state with the provided value before the call of the function.

    Parameters
    ----------
    func : function
        function to be decorated
    seed : int or None
        A seed to be supplied to np.random.seed(). None leaves numpy's
        random state unchanged.

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


def overlaps(df, idx):
    """
    Check if the note at the given index in the given dataframe overlaps any
    other notes in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check for overlaps.

    idx : int
        The index of the note within df that might overlap.

    Returns
    -------
    overlap : boolean
        True if the note overlaps some other note. False otherwise.
    """
    note = df.loc[idx]
    df = df.loc[(df['pitch'] == note.pitch) &
                (df['track'] == note.track) &
                (df.index != idx)]
    offsets = df['onset'] + df['dur']
    overlap = any((note.onset < df['onset'] + df['dur']) &
                  (note.onset + note.dur > df['onset']))
    return overlap


def pre_process(df, sort=False):
    """
    Function which will pre-process a dataframe to be degraded.
    
    Currently, that means resetting the indices to consecutive ints from 0.
    Optionally, this will sort the df (depending on the degradation).
    This function is called automatically by each degradation.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to pre-process.

    sort : boolean
        True to sort the dataframe. Flase to leave the ordering as given.

    Returns
    -------
    df : pd.DataFrame
        The postprocessed dataframe.
    """
    df = df.loc[:, NOTE_DF_SORT_ORDER]
    if sort:
        df = df.sort_values(NOTE_DF_SORT_ORDER)
    df = df.reset_index(drop=True)
    df = df.round().astype(int)
    return df


def post_process(df, sort=True):
    """
    Function which will post-process a degraded dataframe.

    That means optionally sorting it, resetting the indices to be
    consecutive ints starting from 0. All degradations call this
    function after their execution.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to post-process.

    sort : boolean
        True to sort the dataframe. Flase to leave the ordering as given.

    Returns
    -------
    df : pd.DataFrame
        The postprocessed dataframe.
    """
    if sort:
        df = df.sort_values(NOTE_DF_SORT_ORDER)
    df = df.reset_index(drop=True)
    return df


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
        normalized before use.

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
def pitch_shift(excerpt, min_pitch=MIN_PITCH_DEFAULT,
                max_pitch=MAX_PITCH_DEFAULT, distribution=None,
                tries=TRIES_DEFAULT):
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
        pitch. None implies a uniform distribution.

    seed : int
        A seed to be supplied to np.random.seed(). None leaves numpy's
        random state unchanged.

    tries : int
        The number of times to try the degradation before giving up, in the case
        that the degraded excerpt overlaps.


    Returns
    -------
    degraded : pd.DataFrame
        A degradation of the excerpt, with the pitch of one note changed,
        or None if the degradation cannot be performed.
    """
    if len(excerpt) == 0:
        warnings.warn('WARNING: No notes to pitch shift. Returning None.',
                      category=UserWarning)
        return None

    excerpt = pre_process(excerpt)
    orig_dist = distribution

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

    degraded = excerpt

    # Sample a random note
    note_index = valid_notes[randint(len(valid_notes))]
    pitch = degraded.loc[note_index, 'pitch']

    # Shift its pitch
    if distribution is None:
        # Uniform distribution
        if min_pitch != max_pitch or min_pitch != pitch:
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

    # Check if overlaps
    if (overlaps(degraded, note_index) or
        degraded.loc[note_index, 'pitch'] == pitch):
        if tries == 1:
            warnings.warn(TRIES_WARN_MSG)
            return None
        return pitch_shift(excerpt, min_pitch=min_pitch, max_pitch=max_pitch,
                           distribution=orig_dist, tries=tries - 1)

    degraded = post_process(degraded)
    return degraded


@set_random_seed
def time_shift(excerpt, min_shift=MIN_SHIFT_DEFAULT,
               max_shift=MAX_SHIFT_DEFAULT, align_onset=False,
               tries=TRIES_DEFAULT):
    """
    Shift the onset and offset times of one note from the given excerpt,
    leaving its duration unchanged.

    Parameters
    ----------
    excerpt : pd.DataFrame
        An excerpt from a piece of music.

    min_shift : int
        The minimum amount by which the note will be shifted.

    max_shift : int
        The maximum amount by which the note will be shifted.

    align_onset : boolean
        Align the shifted note to the onset time of an existing note
        (within the given shift range).

    seed : int
        A seed to be supplied to np.random.seed(). None leaves numpy's
        random state unchanged.

    tries : int
        The number of times to try the degradation before giving up, in the case
        that the degraded excerpt overlaps.


    Returns
    -------
    degraded : pd.DataFrame
        A degradation of the excerpt, with the timing of one note changed,
        or None if there are no notes that can be changed.
    """
    excerpt = pre_process(excerpt)

    min_shift = max(min_shift, 1)

    onset = excerpt['onset']
    offset = onset + excerpt['dur']
    end_time = offset.max()

    # Shift earlier
    earliest_earlier_onset = (onset - (max_shift - 1)).clip(lower=0)
    latest_earlier_onset = onset - (min_shift - 1)

    # Shift later
    latest_later_onset = onset + (((end_time + 1) - offset)
                                  .clip(upper=max_shift + 1))
    earliest_later_onset = onset + min_shift

    if align_onset:
        # Find ranges which contain a note to align to
        # I couldn't think of a better solution than iterating here.
        # This code checks, for every range, whether at least 1 onset
        # lies within that range.
        onset = pd.Series(onset.unique())
        for i, (eeo, leo, elo, llo) in enumerate(zip(
            earliest_earlier_onset, latest_earlier_onset,
            earliest_later_onset, latest_later_onset)):
            # Go through each range to check there is a valid onset
            earlier_valid = onset.between(eeo, leo - 1).any()
            later_valid = onset.between(elo, llo - 1).any()

            # Close invalid ranges
            if not earlier_valid:
                earliest_earlier_onset.iloc[i] = leo
            if not later_valid:
                earliest_later_onset.iloc[i] = llo

    # Find valid notes
    valid = ((earliest_earlier_onset < latest_earlier_onset) |
             (earliest_later_onset < latest_later_onset))
    valid_notes = list(valid.index[valid])

    if not valid_notes:
        warnings.warn('WARNING: No valid notes to time shift. Returning '
                      'None.', category=UserWarning)
        return None

    # Sample a random note
    index = choice(valid_notes)

    eeo = earliest_earlier_onset[index]
    leo = max(latest_earlier_onset[index], eeo)
    elo = earliest_later_onset[index]
    llo = max(latest_later_onset[index], elo)

    if align_onset:
        valid_onsets = (onset.between(eeo, leo - 1) |
                        onset.between(elo, llo - 1))
        valid_onsets = list(onset[valid_onsets])
        onset = choice(valid_onsets)
    else:
        onset = split_range_sample([(eeo, leo), (elo, llo)])

    degraded = excerpt

    degraded.loc[index, 'onset'] = onset

    # Check if overlaps
    if overlaps(degraded, index):
        if tries == 1:
            warnings.warn(TRIES_WARN_MSG)
            return None
        return time_shift(excerpt, min_shift=min_shift, max_shift=max_shift,
                          align_onset=align_onset, tries=tries - 1)

    degraded = post_process(degraded)
    return degraded


@set_random_seed
def onset_shift(excerpt, min_shift=MIN_SHIFT_DEFAULT,
                max_shift=MAX_SHIFT_DEFAULT, min_duration=MIN_DURATION_DEFAULT,
                max_duration=MAX_DURATION_DEFAULT, align_onset=False,
                align_dur=False, tries=TRIES_DEFAULT):
    """
    Shift the onset time of one note from the given excerpt.

    Parameters
    ----------
    excerpt : df.DataFrame
        An excerpt from a piece of music.

    min_shift : int
        The minimum amount by which the onset time will be changed.

    max_shift : int
        The maximum amount by which the onset time will be changed.

    min_duration : int
        The minimum duration for the resulting note.

    max_duration : int
        The maximum duration for the resulting note.
        (The offset time will never go beyond the current last offset
        in the excerpt.)

    align_onset : boolean
        True to force the shifted onset to lie on an existing onset.

    align_dur : boolean
        True to force the resulting duration to be equal to an existing
        duration.

    seed : int
        A seed to be supplied to np.random.seed(). None leaves numpy's
        random state unchanged.

    tries : int
        The number of times to try the degradation before giving up, in the case
        that the degraded excerpt overlaps.

    Returns
    -------
    degraded : df.DataFrame
        A degradation of the excerpt, with the onset time of one note
        changed, or None if the degradation cannot be performed.
    """
    excerpt = pre_process(excerpt)

    min_shift = max(min_shift, 1)
    min_duration -= 1 # This makes computation below simpler

    onset = excerpt['onset']
    offset = onset + excerpt['dur']
    unique_durs = excerpt['dur'].unique()

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

    if align_onset:
        # Find ranges which contain a note to align to
        # I couldn't think of a better solution than iterating here.
        # This code checks, for every range, whether at least 1 onset
        # lies within that range.
        onset = pd.Series(onset.unique())
        for i, (elo, llo, eso, lso) in enumerate(zip(
            earliest_lengthened_onset, latest_lengthened_onset,
            earliest_shortened_onset, latest_shortened_onset)):
            # Go through each range to check there is a valid onset
            earlier_valid = onset.between(elo, llo - 1)
            later_valid = onset.between(eso, lso - 1)

            if align_dur:
                # Here, align both onset and dur
                resulting_dur = offset.iloc[i] - onset
                dur_valid = resulting_dur.isin(unique_durs)
                earlier_valid = earlier_valid & dur_valid
                later_valid = later_valid & dur_valid

            # Collapse down
            earlier_valid = earlier_valid.any()
            later_valid = later_valid.any()

            # Close invalid ranges
            if not earlier_valid:
                earliest_lengthened_onset.iloc[i] = llo
            if not later_valid:
                earliest_shortened_onset.iloc[i] = lso

    elif align_dur:
        # Here, align_onset is False.
        # Find ranges which contain a duration to align to
        # I couldn't think of a better solution than iterating here.
        # This code checks, for every range, whether at least 1 dur
        # lies within that range.
        durs = pd.Series(unique_durs)
        for i, (elo, llo, lso, eso) in enumerate(zip(
            earliest_lengthened_onset, latest_lengthened_onset,
            latest_shortened_onset, earliest_shortened_onset)):
            # Go through each range to check there is a valid dur
            result = offset[i] - durs
            lengthened_valid = result.between(elo, llo - 1).any()
            shortened_valid = result.between(eso, lso - 1).any()

            # Close invalid ranges
            if not lengthened_valid:
                earliest_lengthened_onset.iloc[i] = llo
            if not shortened_valid:
                earliest_shortened_onset.iloc[i] = lso

    # Find valid notes
    valid = ((earliest_lengthened_onset < latest_lengthened_onset) |
             (earliest_shortened_onset < latest_shortened_onset))
    valid_notes = list(valid.index[valid])

    if not valid_notes:
        warnings.warn('WARNING: No valid notes to onset shift. Returning '
                      'None.', category=UserWarning)
        return None

    # Sample a random note
    index = choice(valid_notes)

    elo = earliest_lengthened_onset[index]
    llo = max(latest_lengthened_onset[index], elo)
    eso = earliest_shortened_onset[index]
    lso = max(latest_shortened_onset[index], eso)

    # Sample onset
    if align_onset:
        valid_onsets = (onset.between(elo, llo - 1) |
                        onset.between(eso, lso - 1))

        if align_dur:
            # Here, align both
            valid_durs = (offset[index] - onset).isin(unique_durs)
            valid_onsets = valid_onsets & valid_durs

        valid_onsets = list(onset[valid_onsets])
        onset = choice(valid_onsets)

    elif align_dur:
        # Align dur but not onset
        onsets = offset[index] - durs
        valid_durs = (onsets.between(elo, llo - 1) |
                      onsets.between(eso, lso - 1))
        valid_durs = list(durs[valid_durs])
        onset = offset[index] - choice(valid_durs)

    else:
        # No alignment
        onset = split_range_sample([(elo, llo), (eso, lso)])

    degraded = excerpt

    degraded.loc[index, 'onset'] = onset
    degraded.loc[index, 'dur'] = offset[index] - onset

    # Check if overlaps
    if overlaps(degraded, index):
        if tries == 1:
            warnings.warn(TRIES_WARN_MSG)
            return None
        return onset_shift(excerpt, min_shift=min_shift, max_shift=max_shift,
                           min_duration=min_duration + 1, # Changed above
                           max_duration=max_duration, align_onset=align_onset,
                           align_dur=align_dur, tries=tries - 1)

    degraded = post_process(degraded)
    return degraded


@set_random_seed
def offset_shift(excerpt, min_shift=MIN_SHIFT_DEFAULT,
                 max_shift=MAX_SHIFT_DEFAULT, min_duration=MIN_DURATION_DEFAULT,
                 max_duration=MAX_DURATION_DEFAULT, align_dur=False,
                 tries=TRIES_DEFAULT):
    """
    Shift the offset time of one note from the given excerpt.

    Parameters
    ----------
    excerpt : df.DataFrame
        An excerpt from a piece of music.

    min_shift : int
        The minimum amount by which the offset time will be changed.

    max_shift : int
        The maximum amount by which the offset time will be changed.

    min_duration : int
        The minimum duration for the resulting note.

    max_duration : int
        The maximum duration for the resulting note.
        (The offset time will never go beyond the current last offset
        in the excerpt.)

    align_dur : boolean
        True to force the resulting duration to be the same as some
        other duration in the given excerpt.

    seed : int
        A seed to be supplied to np.random.seed(). None leaves numpy's
        random state unchanged.

    tries : int
        The number of times to try the degradation before giving up, in the case
        that the degraded excerpt overlaps.


    Returns
    -------
    degraded : df.DataFrame
        A degradation of the excerpt, with the offset time of one note
        changed, or None if the degradation cannot be performed.
    """
    excerpt = pre_process(excerpt)

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

    if align_dur:
        # Find ranges which contain a duration to align to
        # I couldn't think of a better solution than iterating here.
        # This code checks, for every range, whether at least 1 duration
        # lies within that range.
        durs = pd.Series(duration.unique())
        for i, (ssd, lsd, sld, lld) in enumerate(zip(
            shortest_shortened_dur, longest_shortened_dur,
            shortest_lengthened_dur, longest_lengthened_dur)):
            # Go through each range to check there is a valid duration
            shortened_valid = durs.between(ssd, lsd - 1).any()
            lengthened_valid = durs.between(sld, lld - 1).any()

            # Close invalid ranges
            if not shortened_valid:
                shortest_shortened_dur.iloc[i] = lsd
            if not lengthened_valid:
                shortest_lengthened_dur.iloc[i] = lld

    # Find valid notes
    valid = ((shortest_lengthened_dur < longest_lengthened_dur) |
             (shortest_shortened_dur < longest_shortened_dur))
    valid_notes = list(valid.index[valid])

    if not valid_notes:
        warnings.warn('WARNING: No valid notes to offset shift. Returning '
                      'None.', category=UserWarning)
        return None

    # Sample a random note
    index = choice(valid_notes)

    ssd = shortest_shortened_dur[index]
    lsd = max(longest_shortened_dur[index], ssd)
    sld = shortest_lengthened_dur[index]
    lld = max(longest_lengthened_dur[index], sld)

    # Sample new duration
    if align_dur:
        valid_durs = (durs.between(ssd, lsd - 1) |
                      durs.between(sld, lld - 1))
        valid_durs = list(durs[valid_durs])
        duration = choice(valid_durs)
    else:
        duration = split_range_sample([(ssd, lsd), (sld, lld)])

    degraded = excerpt

    degraded.loc[index, 'dur'] = duration

    # Check if overlaps
    if overlaps(degraded, index):
        if tries == 1:
            warnings.warn(TRIES_WARN_MSG)
            return None
        return offset_shift(excerpt, min_shift=min_shift,
                            max_shift=max_shift, min_duration=min_duration,
                            max_duration=max_duration - 1, # Changed above
                            align_dur=align_dur, tries=tries - 1)

    degraded = post_process(degraded)
    return degraded


@set_random_seed
def remove_note(excerpt, tries=TRIES_DEFAULT):
    """
    Remove one note from the given excerpt.

    Parameters
    ----------
    excerpt : df.DataFrame
        An excerpt from a piece of music.

    seed : int
        A seed to be supplied to np.random.seed(). None leaves numpy's
        random state unchanged.

    tries : int
        The number of times to try the degradation before giving up, in the case
        that the degraded excerpt overlaps. This is not used, but we keep it for
        consistency.

    Returns
    -------
    degraded : df.DataFrame
        A degradation of the excerpt, with one note removed, or None if
        the degradations cannot be performed.
    """
    if excerpt.shape[0] == 0:
        warnings.warn('WARNING: No notes to remove. Returning None.',
                      category=UserWarning)
        return None

    degraded = pre_process(excerpt)

    # Sample a random note
    note_index = choice(list(degraded.index))

    # Remove that note
    degraded = degraded.drop(note_index)

    # No need to check for overlap
    degraded = post_process(degraded, sort=False)
    return degraded


@set_random_seed
def add_note(excerpt, min_pitch=MIN_PITCH_DEFAULT, max_pitch=MAX_PITCH_DEFAULT,
             min_duration=MIN_DURATION_DEFAULT,
             max_duration=MAX_DURATION_DEFAULT, align_pitch=False,
             align_time=False, tries=TRIES_DEFAULT):
    """
    Add one note to the given excerpt.

    Parameters
    ----------
    excerpt : df.DataFrame
        An excerpt from a piece of music.

    min_pitch : int
        The minimum pitch at which a note may be added.

    max_pitch : int
        The maximum pitch at which a note may be added.

    min_duration : int
        The minimum duration for the note to be added.

    max_duration : int
        The maximum duration for the added note.
        (The offset time will never go beyond the current last offset
        in the excerpt.)
        
    align_pitch : boolean
        True to force the added note to lie on the same pitch as an
        existing note (if one exists). This ignores the given min and
        max pitches. If excerpt contains only 1 note and align_time
        is True, this is always set to False.
        
    align_time : boolean
        True to force the added note to have the same onset time and
        duration as an existing note (if one exists), though not
        necessarily the same (onset, duration) pair as an existing
        note. If True, this ignores the min and max durations.

    seed : int
        A seed to be supplied to np.random.seed(). None leaves numpy's
        random state unchanged.

    tries : int
        The number of times to try the degradation before giving up, in the case
        that the degraded excerpt overlaps.


    Returns
    -------
    degraded : df.DataFrame
        A degradation of the excerpt, with one note added, or None if
        the degradations cannot be performed.
    """
    excerpt = pre_process(excerpt)

    if len(excerpt) == 0:
        align_pitch = False
        align_time = False

    if len(excerpt) == 1 and align_pitch and align_time:
        align_pitch = False

    end_time = excerpt[['onset', 'dur']].sum(axis=1).max()

    if align_pitch:
        pitch = excerpt['pitch'].between(min_pitch, max_pitch,
                                         inclusive=True)
        pitch = excerpt['pitch'][pitch].unique()
        if len(pitch) == 0:
            warnings.warn("WARNING: No valid aligned pitch in given "
                          "range.", category=UserWarning)
            return None
        pitch = choice(pitch)
    else:
        pitch = randint(min_pitch, max_pitch + 1)

    # Find onset and duration
    if align_time:
        if (min_duration > excerpt['dur'].max() or
            max_duration < excerpt['dur'].min()):
            warnings.warn("WARNING: No valid aligned duration in "
                          "given range.", category=UserWarning)
            return None

        durations = excerpt['dur'].between(min_duration, max_duration,
                                           inclusive=True)
        durations = excerpt['dur'][durations]
        min_dur = durations.min()
        onset = excerpt['onset'].between(0, end_time - min_dur,
                                         inclusive=True)
        onset = choice(excerpt['onset'][onset].unique())
        dur_unique = durations[durations.between(
            min_dur, end_time - onset, inclusive=True)].unique()
        duration = choice(dur_unique)
    elif min_duration > end_time:
        onset = 0
        duration = min_duration
    elif excerpt.shape[0] == 0:
        onset = 0
        duration = randint(min_duration,
                           min(max_duration + 1, sys.maxsize))
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

    # Create and add note
    note = {'pitch': pitch,
            'onset': onset,
            'dur': duration,
            'track': track}

    degraded = excerpt
    degraded = degraded.append(note, ignore_index=True)

    # Check if overlaps
    if overlaps(degraded, degraded.index[-1]):
        if tries == 1:
            warnings.warn(TRIES_WARN_MSG)
            return None
        return add_note(excerpt, min_pitch=min_pitch, max_pitch=max_pitch,
                        min_duration=min_duration, max_duration=max_duration,
                        align_pitch=align_pitch, align_time=align_time,
                        tries=tries - 1)

    degraded = post_process(degraded)
    return degraded


@set_random_seed
def split_note(excerpt, min_duration=MIN_DURATION_DEFAULT, num_splits=1,
               tries=TRIES_DEFAULT):
    """
    Split one note from the excerpt into two or more notes of equal
    duration.

    Parameters
    ----------
    excerpt : df.DataFrame
        An excerpt from a piece of music.

    min_duration : int
        The minimum length for any of the resulting notes.

    num_splits : int
        The number of splits to make in the chosen note. The note will
        be split into (num_splits+1) shorter notes.

    seed : int
        A seed to be supplied to np.random.seed(). None leaves numpy's
        random state unchanged.

    tries : int
        The number of times to try the degradation before giving up, in the case
        that the degraded excerpt overlaps. This is not used, but we keep it for
        consistency.

    Returns
    -------
    degraded : df.DataFrame
        A degradation of the excerpt, with one note split, or None if
        the degradation cannot be performed.
    """
    if excerpt.shape[0] == 0:
        warnings.warn('WARNING: No notes to split. Returning None.',
                      category=UserWarning)
        return None

    excerpt = pre_process(excerpt)

    # Find all splitable notes
    long_enough = excerpt['dur'] >= min_duration * (num_splits + 1)
    valid_notes = list(long_enough.index[long_enough])

    if not valid_notes:
        warnings.warn('WARNING: No valid notes to split. Returning '
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

    degraded = excerpt
    degraded.loc[note_index]['dur'] = int(round(short_duration_float))
    new_df = pd.DataFrame({'onset': onsets,
                           'track': tracks,
                           'pitch': pitches,
                           'dur': durs})
    degraded = degraded.append(new_df, ignore_index=True)

    # No need to check for overlap
    degraded = post_process(degraded)
    return degraded


@set_random_seed
def join_notes(excerpt, max_gap=50, max_notes=20, only_first=False,
               tries=TRIES_DEFAULT):
    """
    Combine two notes of the same pitch and track into one.

    Parameters
    ----------
    excerpt : df.DataFrame
        An excerpt from a piece of music.

    max_gap : int
        The maximum gap length, in ms, for 2 notes to be able to be joined.
        (They must always share the same pitch and track).

    max_notes : int
        The maximum number of notes to join together. This degradation will
        greedily join as many notes together as possible up to this value,
        starting from a randomly chosen note (which may or may not be the
        first note in a sequence, depending on only_first).

    only_first : boolean
        True to always begin joining notes from the first note of a
        sequence of consecutive notes. False will choose a note randomly
        up to the 2nd-to-last note from all valid sequences.

    seed : int
        A seed to be supplied to np.random.seed(). None leaves numpy's
        random state unchanged.

    tries : int
        The number of times to try the degradation before giving up, in the case
        that the degraded excerpt overlaps. This is not used, but we keep it for
        consistency.

    Returns
    -------
    degraded : df.DataFrame
        A degradation of the excerpt, with one note split, or None if
        the degradation cannot be performed.
    """
    if excerpt.shape[0] < 2:
        warnings.warn('WARNING: No notes to join. Returning None.',
                      category=UserWarning)
        return None

    excerpt = pre_process(excerpt, sort=True)

    valid_starts = []
    valid_nexts = []

    for _, track_df in excerpt.groupby('track'):
        for _, pitch_df in track_df.groupby('pitch'):
            if len(pitch_df) < 2:
                continue

            # Get note gaps
            onset = pitch_df['onset']
            offset = onset + pitch_df['dur']
            gap_after = onset.shift(-1) - offset
            gap_after.iloc[-1] = np.inf
            gap_before = gap_after.shift(1)
            gap_before.iloc[0] = np.inf

            # Get valid notes to start joining from
            if only_first:
                valid = (gap_after <= max_gap) & (gap_before > max_gap)
            else:
                valid = gap_after <= max_gap
            valid_starts_this = list(valid.index[valid])
            valid_next_bool = gap_before <= max_gap

            # Get notes to join for each valid start
            for start in valid_starts_this:
                iloc = pitch_df.index.get_loc(start)
                valid_next = []
                for i, v in enumerate(valid_next_bool[iloc + 1:]):
                    if i + 2 > max_notes or not v:
                        break
                    valid_next.append(valid_next_bool.index[iloc + 1 + i])
                valid_nexts.append(valid_next)
            valid_starts.extend(valid_starts_this)

    if not valid_starts:
        warnings.warn('WARNING: No valid notes to join. Returning '
                      'None.', category=UserWarning)
        return None

    index = randint(len(valid_starts))

    start = valid_starts[index]
    nexts = valid_nexts[index]

    degraded = excerpt

    # Extend first note
    degraded.loc[start]['dur'] = (degraded.loc[nexts[-1]]['onset'] +
                                  degraded.loc[nexts[-1]]['dur'] -
                                  degraded.loc[start]['onset'])

    # Drop all following notes note
    degraded = degraded.drop(nexts)

    # No need to check for overlap
    degraded = post_process(degraded)
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


def get_degradations(degradation_list=DEGRADATIONS):
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
