#!/usr/bin/env python
"""Script to measure the errors from a transcription system in order to create
a degraded MIDI dataset with the given proportions of degradations."""
import argparse
import glob
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from mdtk import fileio, formatters
from mdtk.degradations import (
    DEGRADATIONS,
    MAX_GAP_DEFAULT,
    MAX_PITCH_DEFAULT,
    MIN_PITCH_DEFAULT,
    MIN_SHIFT_DEFAULT,
)

FILE_TYPES = ["mid", "pkl", "csv"]


def get_df_excerpt(note_df, start_time, end_time):
    """
    Return an excerpt of the given note_df, with notes cut at the given
    start and end times.

    Parameters
    ----------
    note_df : pd.DataFrame
        The note_df we want to take an excerpt of.

    start_time : int
        The start time for the returned excerpt, in ms, inclusive. Notes
        entirely before this time will be dropped. Notes which onset before
        this time but continue after it will have their onset shifted to
        this time.

    end_time : int
        The end time for the returned excerpt, in ms, exclusive. Notes
        entirely after this time will be dropped. Notes which onset before
        this time but continue after it will have their offset shifted to
        this time. None to enforce no end time.

    Returns
    -------
    note_df : pd.DataFrame
        An excerpt of the notes from the given note_df, within the given
        two times.
    """
    # Make copy so as not to change original values
    note_df = note_df.copy()

    # Move onsets of notes which lie before start (and finish after start)
    need_to_shift = (note_df.onset < start_time) & (
        note_df.onset + note_df.dur > start_time
    )
    shift_amt = start_time - note_df.loc[need_to_shift, "onset"]
    note_df.loc[need_to_shift, "onset"] = start_time
    note_df.loc[need_to_shift, "dur"] -= shift_amt

    # Shorten notes which go past end time
    if end_time is not None:
        need_to_shorten = (note_df.onset < end_time) & (
            note_df.onset + note_df.dur > end_time
        )
        note_df.loc[need_to_shorten, "dur"] = (
            end_time - note_df.loc[need_to_shorten, "onset"]
        )

    # Drop notes which lie outside of bounds
    to_keep = note_df.onset >= start_time
    if end_time is not None:
        to_keep &= note_df.onset < end_time
    note_df = note_df.loc[to_keep]
    return note_df


def load_file(
    filename,
    pr_min_pitch=MIN_PITCH_DEFAULT,
    pr_max_pitch=MAX_PITCH_DEFAULT,
    pr_time_increment=40,
):
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

    if ext == ".mid":
        return fileio.midi_to_df(filename)

    if ext == ".csv":
        return fileio.csv_to_df(filename)

    if ext == ".pkl":
        with open(filename, "rb") as file:
            pkl = pickle.load(file)

        piano_roll = pkl["piano_roll"]

        if piano_roll.shape[1] == (pr_min_pitch - pr_max_pitch + 1):
            # Normal piano roll only -- no onsets
            note_pr = piano_roll.astype(int)
            onset_pr = (np.roll(note_pr, 1, axis=0) - note_pr) == -1
            onset_pr[0] = note_pr[0]
            onset_pr = onset_pr.astype(int)

        elif piano_roll.shape[1] == 2 * (pr_min_pitch - pr_max_pitch + 1):
            # Piano roll with onsets
            note_pr = piano_roll[:, : piano_roll.shape[1] / 2].astype(int)
            onset_pr = piano_roll[:, piano_roll.shape[1] / 2 :].astype(int)

        else:
            raise ValueError(
                "Piano roll dimension 2 size ("
                f"{piano_roll.shape[1]}) must be equal to 1 or 2"
                f" times the given pitch range [{pr_min_pitch} - "
                f"{pr_max_pitch}] = "
                f"{pr_min_pitch - pr_max_pitch + 1}"
            )

        piano_roll = np.vstack((note_pr, onset_pr))
        return formatters.double_pianoroll_to_df(
            piano_roll,
            min_pitch=pr_min_pitch,
            max_pitch=pr_max_pitch,
            time_increment=pr_time_increment,
        )

    raise NotImplementedError(f"Extension {ext} not supported.")


def merge_on_pitch(gt_df, trans_df, offset=True):
    """
    Merge the given ground truth and transcribed dfs on pitch, with
    corresponding suffixes, and possibly offset columns.

    Parameters
    ----------
    gt_df : pd.DataFrame
        The ground truth data frame.

    trans_df : pd.DataFrame
        The transcription data frame.

    offset : boolean
        Calculate offset columns pre-merge.

    Results
    -------
    merge_df : pd.DataFrame
        The gt and trans DataFrames, merged on equal pitches, with index
        columns added for each pre-merge, and _gt and _trans suffixes added
        to the resulting columns. If offset is True, offset columns are
        calculated pre-merge.
    """
    # This both creates a copy and creates an index column which will be
    # retained in the merge
    gt_df = gt_df.reset_index()
    trans_df = trans_df.reset_index()

    # Pre-calculate offset time once
    if offset:
        gt_df["offset"] = gt_df.onset + gt_df.dur
        trans_df["offset"] = trans_df.onset + trans_df.dur

    # Merge notes with equal pitch -- keep all pairs
    return trans_df.reset_index().merge(
        gt_df.reset_index(), on="pitch", suffixes=("_trans", "_gt")
    )


def get_correct_notes(
    gt_df, trans_df, max_onset_err=MIN_SHIFT_DEFAULT, max_offset_err=MIN_SHIFT_DEFAULT
):
    """
    Get lists of the correctly transcribed notes' indices.

    Parameters
    ----------
    gt_df : pd.DataFrame
        The ground truth data frame.

    trans_df : pd.DataFrame
        The transcription data frame.

    max_onset_err : int
        The maximum error for 2 onsets to be considered simultaneous.

    max_offset_err : int
        The maximum error for 2 offsets to be considered simultaneous.

    Returns
    -------
    correct_gt : list(int)
        A list of the indices of the correctly transcribed ground truth
        notes.

    correct_trans : list(int)
        A list of the indices of the correctly transcribed transcribed
        notes.
    """
    correct_gt = []
    correct_trans = []

    # Merge on pitch
    merged_df = merge_on_pitch(gt_df, trans_df, offset=True)

    # Keep only notes close enough at onset and offset
    merged_df["onset_diff"] = (merged_df.onset_trans - merged_df.onset_gt).abs()
    onset_close = merged_df.onset_diff < max_onset_err
    offset_close = (
        merged_df.offset_trans - merged_df.offset_gt
    ).abs() <= max_offset_err
    merged_df = merged_df.loc[onset_close & offset_close]

    while len(merged_df) > 0:
        # Keep only match closest to correct onset
        matched_notes = merged_df.loc[
            merged_df.groupby("index_gt")["onset_diff"].idxmin()
        ]

        # Remove duplicate trans note matches
        matched_notes = matched_notes.loc[~matched_notes.index_trans.duplicated()]

        # Save matches
        correct_gt.extend(list(matched_notes.index_gt))
        correct_trans.extend(list(matched_notes.index_trans))

        # Remove saved from merged_df for next pass (in case of duplicates)
        trans_matched = merged_df.index_trans.isin(matched_notes.index_trans)
        gt_matched = merged_df.index_gt.isin(matched_notes.index_gt)
        merged_df = merged_df.loc[~trans_matched & ~gt_matched]

    return correct_gt, correct_trans


def get_shifts(
    gt_df,
    trans_df,
    max_onset_err=MIN_SHIFT_DEFAULT,
    max_offset_err=MIN_SHIFT_DEFAULT,
    max_dur_err=MIN_SHIFT_DEFAULT,
):
    """
    Get the shift degradations (onset, offset, time, pitch) of the
    given ground truth and transcription.

    Parameters
    ----------
    gt_df : pd.DataFrame
        The ground truth data frame.

    trans_df : pd.DataFrame
        The transcription data frame.

    max_onset_err : int
        The maximum error for 2 onsets to be considered simultaneous.

    max_offset_err : int
        The maximum error for 2 offsets to be considered simultaneous.

    max_dur_err : int
        The maximum error for 2 durations to be considered of equal length.

    Returns
    -------
    shift_df : pd.DataFrame
        A DataFrame containing information about each found shift, which
        can be used to count each degradation or to calculate parameters.
    """
    shifts = []

    # Save fully merged df
    merged_df = merge_on_pitch(gt_df, trans_df, offset=True)
    merged_df["onset_diff"] = (merged_df.onset_trans - merged_df.onset_gt).abs()
    merged_df["offset_diff"] = (merged_df.offset_trans - merged_df.offset_gt).abs()
    merged_df["dur_diff"] = (merged_df.dur_trans - merged_df.dur_gt).abs()

    # First, check for time offset shifts
    # Onset is close enough
    match_df = merged_df.loc[merged_df.onset_diff <= max_onset_err]
    # Keep only match closest to correct onset
    match_df = match_df.loc[
        match_df.index
        == match_df.groupby("index_gt")["onset_diff"].idxmin()[match_df.index_gt]
    ]
    match_df["deg_type"] = "offset_shift"
    shifts.extend(match_df.to_dict("records"))

    # Filter offset shifts out of base dfs
    merged_df = merged_df.loc[
        ~(
            merged_df.index_trans.isin(match_df.index_trans)
            | merged_df.index_gt.isin(match_df.index_gt)
        )
    ].copy()
    gt_df = gt_df.drop(index=match_df.index_gt)
    trans_df = trans_df.drop(index=match_df.index_trans)

    # Second, check for onset shifts
    # Offset is close enough
    match_df = merged_df.loc[merged_df.offset_diff <= max_offset_err]
    # Keep only match closest to correct offset
    match_df = match_df.loc[
        match_df.index
        == match_df.groupby("index_gt")["offset_diff"].idxmin()[match_df.index_gt]
    ]
    match_df["deg_type"] = "onset_shift"
    shifts.extend(match_df.to_dict("records"))

    # Filter onset shifts out of base dfs
    merged_df = merged_df.loc[
        ~(
            merged_df.index_trans.isin(match_df.index_trans)
            | merged_df.index_gt.isin(match_df.index_gt)
        )
    ].copy()
    gt_df = gt_df.drop(index=match_df.index_gt)
    trans_df = trans_df.drop(index=match_df.index_trans)

    # Third, check for time shifts
    # Dur is close enough
    match_df = merged_df.loc[merged_df.dur_diff <= max_dur_err]
    # Shift is small enough (smaller than gt note duration)
    match_df = match_df.loc[match_df.onset_diff <= match_df.dur_gt]
    # Keep only match shortest shift
    match_df = match_df.loc[
        match_df.index
        == match_df.groupby("index_gt")["onset_diff"].idxmin()[match_df.index_gt]
    ]
    match_df["deg_type"] = "time_shift"
    shifts.extend(match_df.to_dict("records"))

    # Filter time shifts out of base dfs (merged is unused past here)
    gt_df = gt_df.drop(index=match_df.index_gt)
    trans_df = trans_df.drop(index=match_df.index_trans)

    # Fourth, check for pitch shifts
    gt_df["offset"] = gt_df.onset + gt_df.dur
    trans_df["offset"] = trans_df.onset + trans_df.dur

    # Looping is necesary because we only get 1 gt_note per trans note,
    # Although, each trans_note may be associated with multiple gt notes
    # at each iteration.
    while len(gt_df) and len(trans_df) > 0:
        # Find onset time closest to each gt note
        gt_df["closest_onset_idx"] = gt_df.apply(
            lambda x: (trans_df.onset - x.onset).abs().idxmin(), axis=1
        )
        gt_df["closest_onset"] = (
            trans_df.loc[gt_df.closest_onset_idx, "onset"].to_numpy() - gt_df.onset
        ).abs()
        # Here, gt_df will eventually become empty (to exit while loop)
        gt_df = gt_df.loc[gt_df.closest_onset <= max_onset_err]

        # Sort by closest onset, and then only take first of each trans_idx
        gt_df = gt_df.sort_values(by="closest_onset")
        pitch_df = gt_df.drop_duplicates(subset="closest_onset_idx")

        # Add to pitch shifts and also_offset
        if len(pitch_df) > 0:
            dict_of_lists = {
                "index_gt": pitch_df.index.values,
                "index_trans": pitch_df.closest_onset_idx.values,
                "pitch_gt": gt_df.loc[pitch_df.index, "pitch"].values,
                "pitch_trans": trans_df.loc[pitch_df.closest_onset_idx, "pitch"].values,
                "deg_type": ["pitch_shift"] * len(pitch_df),
            }
            shifts.extend(
                [
                    {key: dict_of_lists[key][i] for key in dict_of_lists}
                    for i in range(len(dict_of_lists["index_gt"]))
                ]
            )

            offset_diff = (
                pitch_df.offset - trans_df.loc[pitch_df.closest_onset_idx, "offset"]
            )
            also_offset_df = pitch_df.loc[offset_diff.abs() > max_offset_err]

            # Pre-indexed helper dfs to make calculation faster
            tmp_trans_df = trans_df.loc[also_offset_df.closest_onset_idx]
            tmp_gt_df = gt_df.loc[also_offset_df.index]

            dict_of_lists = {
                "index_gt": also_offset_df.index.values,
                "index_trans": also_offset_df.closest_onset_idx.values,
                "pitch_gt": tmp_gt_df["pitch"].values,
                "pitch_trans": tmp_trans_df["pitch"].values,
                "onset_gt": tmp_gt_df["onset"].values,
                "onset_trans": tmp_trans_df["onset"].values,
                "dur_gt": tmp_gt_df["dur"].values,
                "dur_trans": tmp_trans_df["dur"].values,
                "offset_gt": (tmp_gt_df["dur"] + tmp_gt_df["onset"]).values,
                "offset_trans": (tmp_trans_df["dur"] + tmp_trans_df["onset"]).values,
                "deg_type": ["pitch_shift"] * len(also_offset_df),
            }
            shifts.extend(
                [
                    {key: dict_of_lists[key][i] for key in dict_of_lists}
                    for i in range(len(dict_of_lists["index_gt"]))
                ]
            )

            # Remove matches from gt_df and trans_df
            gt_df = gt_df.drop(index=pitch_df.index)
            trans_df = trans_df.drop(index=pitch_df.closest_onset_idx)

    return pd.DataFrame(shifts)


def update_shift_params(shift_df, params):
    """
    Get parameters for the *_shift degradations given the found shift errors
    and the ground truth and transcription excerpts.

    Parameters
    ----------
    shift_df : pd.DataFrame
        A DataFrame containing information about each found shift, which
        can be used to count each degradation or to calculate parameters.

    params : Dict(str -> float)
        A dictionary in which to store, update, and return (in place) parameters
        for the degradation.
    """

    def update_value(params_dict, key, update_func, new_value, cast_func=None):
        """
        Update a dictionary entry to either a new value, or a given function applied
        to the old and new values.

        Parameters
        ----------
        params_dict : dict
            The dictionary to be updated.

        key : string
            The key whose value should be updated in the dictionary.

        update_func : Func
            A function taking 2 arguments. If the key already exists in the dictionary,
            its new value will be this function called with the old value and
            new_value.

        new_value : Any
            If the key doesn't exist in the dictionary, it will be set to this.

        cast_func : Func
            A function that can be used to cast the result to a desired type.
        """
        params_dict[key] = (
            update_func(params_dict[key], new_value)
            if key in params_dict
            else new_value
        )
        if cast_func is not None:
            params_dict[key] = cast_func(params_dict[key])

    # Onset shift
    onset_df = shift_df.loc[shift_df.deg_type == "onset_shift"]
    if len(onset_df) > 0:
        shift_amt = (onset_df["onset_gt"] - onset_df["onset_trans"]).abs()
        update_value(params, "onset_shift__min_shift", min, shift_amt.min(), int)
        update_value(params, "onset_shift__max_shift", max, shift_amt.max(), int)
        update_value(
            params, "onset_shift__min_duration", min, onset_df.dur_trans.min(), int
        )
        update_value(
            params, "onset_shift__max_duration", max, onset_df.dur_trans.max(), int
        )

    # Offset shift
    offset_df = shift_df.loc[shift_df.deg_type == "offset_shift"]
    if len(offset_df) > 0:
        shift_amt = (
            (offset_df["onset_gt"] + offset_df["dur_gt"])
            - (offset_df["onset_trans"] + offset_df["dur_trans"])
        ).abs()
        update_value(params, "offset_shift__min_shift", min, shift_amt.min(), int)
        update_value(params, "offset_shift__max_shift", max, shift_amt.max(), int)
        update_value(
            params, "offset_shift__min_duration", min, offset_df.dur_trans.min(), int
        )
        update_value(
            params, "offset_shift__max_duration", max, offset_df.dur_trans.max(), int
        )

    # Time shift
    time_df = shift_df.loc[shift_df.deg_type == "time_shift"]
    if len(time_df) > 0:
        shift_amt = (time_df["onset_gt"] - time_df["onset_trans"]).abs()
        update_value(params, "time_shift__min_shift", min, shift_amt.min(), int)
        update_value(params, "time_shift__max_shift", max, shift_amt.max(), int)

    # Pitch shift
    pitch_df = shift_df.loc[shift_df.deg_type == "pitch_shift"]
    if len(pitch_df) > 0:
        update_value(
            params, "pitch_shift__min_pitch", min, pitch_df.pitch_trans.min(), int
        )
        update_value(
            params, "pitch_shift__max_pitch", max, pitch_df.pitch_trans.max(), int
        )

        # TODO: How to do pitch distribution?


def get_joins(gt_df, trans_df, max_gap=MAX_GAP_DEFAULT):
    """
    Find any notes in the ground truth which have been joined in the
    transcription.

    Parameters
    ----------
    gt_df : pd.DataFrame
        The ground truth data frame.

    trans_df : pd.DataFrame
        The transcription data frame.

    max_gap : int
        The maximum allowed gap between notes to be joined, in ms.

    Returns
    -------
    pre_joined_notes : list(list(pd.Index))
        A list of the notes that have been joined. Each element in
        pre_joined_notes represents a list of notes that have been joined
        together in the given transcription.

    post_joined_notes : list(pd.Index)
        A list of the notes (in trans_df) resulting from each join
        in pre_joined_notes.

    shift_onset : list(boolean)
        A list of bools indicating whether the onset of the corresponding
        join has been shifted (in addition to the join).

    shift_offset : list(boolean)
        A list of bools indicating whether the offset of the corresponding
        join has been shifted (in addition to the join).
    """
    pre_joined_notes = []
    post_joined_notes = []
    shift_onset = []
    shift_offset = []

    # Merge on pitch
    merged_df = merge_on_pitch(gt_df, trans_df, offset=True)

    # Save only rows where notes overlap enough
    # Must overlap at least half of min(max gap, gt_duration)
    overlap_start = merged_df[["onset_trans", "onset_gt"]].max(axis=1)
    overlap_end = merged_df[["offset_trans", "offset_gt"]].min(axis=1)
    merged_df["overlap_length"] = overlap_end - overlap_start
    merged_df = merged_df.loc[
        merged_df.overlap_length >= 0.5 * merged_df.dur_gt.clip(upper=max_gap)
    ]

    # Keep only trans notes with multiple overlapping gt notes
    merged_df = merged_df.loc[merged_df.index_trans.duplicated(keep=False)].copy()

    # Save only rows where consecutive gt notes have small enough gap
    # (Also save last rows of each trans note)
    valid_gap = (merged_df.onset_gt.shift(-1) - merged_df.offset_gt) <= max_gap
    last_trans_note = merged_df.index_trans != merged_df.index_trans.shift(-1)
    valid_note = valid_gap | last_trans_note

    # This line counts the number of Falses before each row
    # This allows us to find consecutive Trues based on having the same value here
    # Note that the last False before a True will be included in the cumsum group
    # The second line filters the Falses out, leaving only the Trues
    merged_df["invalid_count"] = (~valid_note).cumsum()
    merged_df = merged_df.loc[valid_note].copy()

    # Now, group by trans note, and find each one's largest chunk of True
    merged_df["largest_group_invalid_count"] = merged_df.groupby("index_trans")[
        "invalid_count"
    ].transform(lambda x: x.value_counts().idxmax())
    merged_df["largest_group_size"] = merged_df.groupby("index_trans")[
        "invalid_count"
    ].transform(lambda x: x.value_counts().max())

    # Select each one's largest group, if of size > 1
    merged_df = merged_df.loc[
        (merged_df.invalid_count == merged_df.largest_group_invalid_count)
        & (merged_df.largest_group_size > 1)
    ].copy()

    # Check for onset/offset shifts
    merged_df["onset_close"] = (
        merged_df.onset_trans - merged_df.onset_gt
    ).abs() <= max_gap
    merged_df["offset_close"] = (
        merged_df.offset_trans - merged_df.offset_gt
    ).abs() <= max_gap

    # Generate output lists
    for trans_id, trans_note_df in merged_df.groupby("index_trans"):
        pre_joined_notes.append(list(trans_note_df.index_gt))
        post_joined_notes.append(trans_id)
        shift_onset.append(not trans_note_df.iloc[0].onset_close)
        shift_offset.append(not trans_note_df.iloc[-1].offset_close)

    return pre_joined_notes, post_joined_notes, shift_onset, shift_offset


def get_splits(gt_df, trans_df, max_gap=MAX_GAP_DEFAULT):
    """
    Find any notes in the ground truth which have been split in the
    transcription.

    This is equivalent to calling get_joins with gt_df and trans_df
    swapped.

    Parameters
    ----------
    gt_df : pd.DataFrame
        The ground truth data frame.

    trans_df : pd.DataFrame
        The transcription data frame.

    max_gap : int
        The maximum allowed gap between notes post split, in ms.

    Returns
    -------
    pre_split_notes : list(pd.Index)
        A list of the notes that have been split. Each element in
        pre_joined_notes represents a note that has been split in the
        given transcription.

    post_split_notes : list(list(pd.Index))
        A list of the notes (in trans_df) resulting from each split
        in pre_split_notes.

    shift_onset : list(boolean)
        A list of bools indicating whether the onset of the corresponding
        join has been shifted (in addition to the split).

    shift_offset : list(boolean)
        A list of bools indicating whether the offset of the corresponding
        join has been shifted (in addition to the split).
    """
    # Split is exactly reverse of a join
    post_split_notes, pre_split_notes, shift_onset, shift_offset = get_joins(
        trans_df, gt_df, max_gap=max_gap
    )

    return pre_split_notes, post_split_notes, shift_onset, shift_offset


def get_excerpt_degs(gt_excerpt, trans_excerpt, params):
    """
    Get the count of each degradation given a ground truth excerpt and a
    transcribed excerpt.

    Parameters
    ----------
    gt_excerpt : pd.DataFrame
        The ground truth data frame.

    trans_excerpt : pd.DataFrame
        The corresponding transcribed dataframe.

    params : dict
        A dictionary to update, store, and return (in place) parameters for
        the different degradations.

    Returns
    -------
    degs : np.array(float)
        The estimated count of each degradation in this transcription, in the
        order given by mdtk.degradations.DEGRADATIONS.
    """
    deg_counts = np.zeros(len(DEGRADATIONS))

    # Remove equal notes
    correct_gt, correct_trans = get_correct_notes(gt_excerpt, trans_excerpt)
    gt_excerpt = gt_excerpt.drop(index=correct_gt)
    trans_excerpt = trans_excerpt.drop(index=correct_trans)

    # Check for joins
    pre_joined_notes, post_joined_notes, shift_onset, shift_offset = get_joins(
        gt_excerpt, trans_excerpt
    )
    deg_counts[list(DEGRADATIONS).index("join_notes")] = len(pre_joined_notes)
    deg_counts[list(DEGRADATIONS).index("onset_shift")] = sum(shift_onset)
    deg_counts[list(DEGRADATIONS).index("offset_shift")] = sum(shift_offset)
    gt_excerpt = gt_excerpt.drop(
        index=[idx for join in pre_joined_notes for idx in join]
    )
    trans_excerpt = trans_excerpt.drop(index=post_joined_notes)

    # Check for splits
    pre_split_notes, post_split_notes, shift_onset, shift_offset = get_splits(
        gt_excerpt, trans_excerpt
    )
    deg_counts[list(DEGRADATIONS).index("split_note")] = len(pre_split_notes)
    deg_counts[list(DEGRADATIONS).index("onset_shift")] += sum(shift_onset)
    deg_counts[list(DEGRADATIONS).index("offset_shift")] += sum(shift_offset)
    gt_excerpt = gt_excerpt.drop(index=pre_split_notes)
    trans_excerpt = trans_excerpt.drop(
        index=[idx for split in post_split_notes for idx in split]
    )

    # Shift degredation estimation (onset, offset, time, pitch)
    shift_df = get_shifts(gt_excerpt, trans_excerpt)
    total_shifts = 0
    if len(shift_df) > 0:
        for deg_type in ["onset_shift", "offset_shift", "time_shift", "pitch_shift"]:
            deg_counts[list(DEGRADATIONS).index(deg_type)] += len(
                shift_df.loc[shift_df.deg_type == deg_type]
            )

        update_shift_params(shift_df, params)

        total_shifts = len(set(shift_df.index_gt))  # Only count unique gt notes

    # Remainder are all adds and removes
    deg_counts[list(DEGRADATIONS).index("add_note")] = len(trans_excerpt) - total_shifts
    deg_counts[list(DEGRADATIONS).index("remove_note")] = len(gt_excerpt) - total_shifts

    return deg_counts


def get_proportions(
    gt, trans, params, trans_start=0, trans_end=None, length=5000, min_notes=10
):
    """
    Get the proportion of each degradation given a ground truth file and
    its transcription.

    This measures the expected count of each degradation given a random
    excerpt from the ground truth (that is NOT clean). And probability
    that a random excerpt will be clean.

    Parameters
    ----------
    gt : string
        The filename of a ground truth musical score.

    trans : string
        The filename of a transciption of the given ground truth.

    params : dict
        A dictionary to store and return (in place) parameter settings
        for the degradations.

    trans_start : int
        The starting time of the transcription, in ms.

    trans_end : int
        The ending time of the transcription, in ms.

    length : int
        The length of the excerpts to grab in ms (plus sustains).

    min_notes : int
        The minimum number of notes required for an excerpt to be valid.

    Returns
    -------
    proportions : list(float)
        The expected count of each degradation present in a random excerpt
        from the given ground truth (that is NOT correctly transcribed), in
        the order given by mdtk.degradations.DEGRADATIONS.

    clean : float
        The estimated probability that a random excerpt from the given
        ground truth will have no errors in the transcription.
    """
    num_excerpts = 0
    deg_counts = np.zeros(len(DEGRADATIONS))
    clean_count = 0

    gt_df = load_file(gt)
    trans_df = load_file(trans)

    # Enforce transcription bounds
    gt_df = get_df_excerpt(gt_df, trans_start, trans_end)
    if trans_start != 0:
        gt_df.onset -= trans_start

    # Calculate latest end time (if else solves nan issue)
    if len(gt_df) == 0:
        end_time = (trans_df.onset + trans_df.dur).max()
    elif len(trans_df) == 0:
        end_time = (gt_df.onset + gt_df.dur).max()
    else:
        end_time = max(
            (gt_df.onset + gt_df.dur).max(), (trans_df.onset + trans_df.dur).max()
        )
    # Take each excerpt from time 0 until the end
    for excerpt_start in range(0, end_time, length):
        excerpt_end = min(excerpt_start + length, end_time)
        gt_excerpt = get_df_excerpt(gt_df, excerpt_start, excerpt_end)
        trans_excerpt = get_df_excerpt(trans_df, excerpt_start, excerpt_end)

        # Check for validity
        if len(gt_excerpt) < min_notes and len(trans_excerpt) < min_notes:
            logging.warning(
                f"Skipping excerpt {gt} for too few notes. "
                f"Time range = [{excerpt_start}, {excerpt_end}). "
                f"Try lowering the minimum note count --min-notes "
                f"(currently {min_notes}), or "
                "ignore this if it is just due to a song length "
                "not being divisible by the --excerpt-length "
                f"(currently {length})."
            )
            continue

        num_excerpts += 1
        excerpt_degs = get_excerpt_degs(gt_excerpt, trans_excerpt, params)
        deg_counts += excerpt_degs
        if np.sum(excerpt_degs) == 0:
            clean_count += 1

    # Divide number of errors by the number of possible excerpts
    if num_excerpts - clean_count == 0:
        proportions = deg_counts
    else:
        proportions = deg_counts / (num_excerpts - clean_count)
    clean = clean_count / num_excerpts if num_excerpts > 0 else 0
    return proportions, clean


def parse_args(args_input=None):
    parser = argparse.ArgumentParser(
        description="Measure errors from a "
        "transcription error in order to make "
        "a degraded MIDI dataset with the measure"
        " proportion of each degration."
    )

    parser.add_argument(
        "--json",
        help="The file to write the degradation config json data out to.",
        default="config.json",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        help="Search the given --gt and "
        "--trans directories recursively. The directory structures"
        " in each don't have to be identical, but corresponding "
        "files must still be uniquely named.",
        action="store_true",
    )

    parser.add_argument(
        "--gt",
        help="The directory which contains the ground "
        "truth musical scores or piano rolls.",
        required=True,
    )
    parser.add_argument(
        "--gt_ext",
        choices=FILE_TYPES,
        default=None,
        help="Restrict the file type for the ground truths.",
    )

    parser.add_argument(
        "--trans",
        help="The directory which contains the transcriptions.",
        required=True,
    )
    parser.add_argument(
        "--trans_ext",
        choices=FILE_TYPES,
        default=None,
        help="Restrict the file type for the transcriptions.",
    )

    # Pianoroll specific args
    parser.add_argument(
        "--pr-min-pitch", type=int, default=21, help="Minimum pianoroll pitch."
    )
    parser.add_argument(
        "--pr-max-pitch", type=int, default=108, help="Maximum pianoroll pitch."
    )

    # Transcription doesn't have same time basis as ground truth
    parser.add_argument(
        "--trans_start",
        type=int,
        default=0,
        help="What time"
        " the transcription starts, in ms. Notes before this "
        "in the gt will be ignored, and all transcribed notes "
        "will be shifted forward by this amount.",
    )
    parser.add_argument(
        "--trans_end",
        type=int,
        default=None,
        help="What time"
        "the transcription ends, in ms (if any). Notes after "
        "this in the gt will be ignored, and notes still on "
        "will be cut at this time.",
    )

    # Excerpt arguments
    parser.add_argument(
        "--excerpt-length",
        metavar="ms",
        type=int,
        help="The length of the excerpt (in ms) to take from "
        "each piece. The excerpt will start on a note onset "
        "and include all notes whose onset lies within this "
        "number of ms after the first note.",
        default=5000,
    )
    parser.add_argument(
        "--min-notes",
        metavar="N",
        type=int,
        default=10,
        help="The minimum number of notes required for an excerpt to be valid.",
    )
    args = parser.parse_args(args=args_input)
    return args


if __name__ == "__main__":
    args = parse_args()

    # Get allowed file extensions
    trans_ext = [args.trans_ext] if args.trans_ext is not None else FILE_TYPES
    gt_ext = [args.gt_ext] if args.gt_ext is not None else FILE_TYPES

    if args.recursive:
        args.trans = os.path.join(args.trans, "**")
        args.gt = os.path.join(args.gt, "**")

    trans = []
    for ext in trans_ext:
        trans.extend(
            glob.glob(os.path.join(args.trans, "*." + ext), recursive=args.recursive)
        )

    proportion = []
    clean_prop = []

    for file in tqdm(trans):
        basename = os.path.splitext(os.path.basename(file))[0]

        # Find gt file
        gt_list = []
        for ext in gt_ext:
            gt_list.extend(
                glob.glob(
                    os.path.join(args.gt, basename + "." + ext),
                    recursive=args.recursive,
                )
            )

        if len(gt_list) == 0:
            logging.warning(
                f"No ground truth found for transcription {file}. Check"
                " that the file extension --gt_ext is correct (or not "
                "given), and the dir --gt is correct. Searched for file"
                f" {basename}.{gt_ext} in dir {args.gt}."
            )
            continue
        elif len(gt_list) > 1:
            logging.warning(
                f"Multiple ground truths found for transcription {file}:"
                f"{gt_list}. Defaulting to the first one. Try narrowing "
                "down extensions with --gt_ext."
            )
        gt = gt_list[0]

        params = {}

        prop, clean = get_proportions(
            gt,
            file,
            params,
            trans_start=args.trans_start,
            trans_end=args.trans_end,
            length=args.excerpt_length,
            min_notes=args.min_notes,
        )
        if sum(prop) > 0:
            proportion.append(prop)
        if sum(prop) + clean > 0:
            clean_prop.append(clean)

    # We want the mean deg_count per file
    proportion = np.mean(proportion, axis=0)
    clean = np.mean(clean_prop)
    params.update({"degradation_dist": proportion.tolist(), "clean_prop": clean})

    with open(args.json, "w") as file:
        json.dump(
            params,
            file,
            indent=4,
        )
