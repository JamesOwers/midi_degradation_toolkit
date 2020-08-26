"""Module containing methods to evaluate performance on Error Correction"""
import numpy as np
from mir_eval.transcription import precision_recall_f1_overlap


def ErrorDetection(outputs, targets):
    """
    Evaluate Error Detection, given a list of outputs and targets.

    Parameters
    ----------
    outputs : np.array(int)
        A list of binary outputs for the task.

    targets : np.array(int)
        A list of binary targets for the task.

    Returns
    -------
    rev_p : float
        The reverse precision (treating 0 as 1 and 1 as 0) given the
        outputs and targets.

    rev_r : float
        The reverse recall (treating 0 as 1 and 1 as 0) given the
        outputs and targets.

    rev_fm : float
        The reverse F-measure (treating 0 as 1 and 1 as 0) given the
        outputs and targets.
    """
    rev_tp = np.sum(np.logical_and(outputs == 0, targets == 0))
    rev_fp = np.sum(1 - outputs) - rev_tp
    rev_fn = np.sum(1 - targets) - rev_tp

    rev_p, rev_r, rev_fm = get_f1(rev_tp, rev_fp, rev_fn)
    return rev_p, rev_r, rev_fm


def ErrorClassification(outputs, targets):
    """
    Evaluate Error Classification, given a list of outputs and targets.

    Parameters
    ----------
    outputs : np.array(int)
        A list of outputs for the task.

    targets : np.array(int)
        A list of targets for the task.

    Returns
    -------
    acc : float
        The accuracy of the outputs given the targets.
    """
    return np.sum(outputs == targets) / len(outputs)


def ErrorLocation(outputs, targets):
    """
    Evaluate Error Location, given a list of outputs and targets.

    Parameters
    ----------
    outputs : np.ndarray(int)
        A (data_points X frames) array of binary outputs for the task.

    targets : np.array(int)
        A (data_points X frames) array of binary targets for the task.

    Returns
    -------
    p : float
        The precision given the outputs and targets.

    r : float
        The recall given the outputs and targets.

    fm : float
        The  F-measure given the outputs and targets.
    """
    tp = np.sum(np.logical_and(outputs == 1, targets == 1))
    fp = np.sum(outputs) - tp
    fn = np.sum(targets) - tp

    p, r, fm = get_f1(tp, fp, fn)
    return p, r, fm


def ErrorCorrection(corrected_df, degraded_df, clean_df, time_increment=40):
    """
    Get the helpfulness of an excerpt, given the degraded and clean versions.
    This is the evaluation metric for the Error Correction task, which can
    also be called by the alias 'helpfulness'.

    Parameters
    ----------
    corrected_df : pd.DataFrame
        The data frame of the corrected excerpt as output by a model.

    degraded_df : pd.DataFrame
        The degraded excerpt given as input to the model.

    clean_df : pd.DataFrame
        The clean excerpt expected as output.

    time_increment : int
        The length of a frame to use when performing evaluations.

    Returns
    -------
    helpfulness : float
        The helpfulness of the given correction, defined as follows:
        -- First, take the mean of the note-based and frame-based F-measures
           of corrected and degraded.
        -- If degraded's score is 1, the mean of note-based and frame-based
           F-measures of corrected vs clean is the output.
        -- Else, using 0.0 == 0.0, degraded == 0.5 and clean == 1.0 as anchor
           points, place corrected's score on that scale.

    f_measure : float
        The average of the degraded_df's framewise F-measure and its notewise
        F-measure, when compared to the clean_df.
    """
    corrected_fm = get_combined_fmeasure(
        corrected_df, clean_df, time_increment=time_increment
    )

    # Quick exit on border cases
    if corrected_fm == 0 or corrected_fm == 1 or degraded_df.equals(clean_df):
        return corrected_fm, corrected_fm

    degraded_fm = get_combined_fmeasure(
        degraded_df, clean_df, time_increment=time_increment
    )

    if corrected_fm < degraded_fm:
        h = 0.5 * corrected_fm / degraded_fm
    else:
        h = 1 - 0.5 * (1 - corrected_fm) / (1 - degraded_fm)

    return h, corrected_fm


helpfulness = ErrorCorrection


def get_combined_fmeasure(df, gt_df, time_increment=40):
    """
    Get the combined framewise and notewise F-measure of a dtaframe,
    given a ground truth dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe whose F-measure to return.

    gt_df : pd.DataFrame
        The ground truth dataframe.

    time_increment : int
        The length of a frame to use when performing evaluations.

    Returns
    -------
    f_measure : float
        The combined f_measure of the given dataframe. The mean of its
        framewise and notewise F-measures.
     """
    fw = get_framewise_f_measure(df, gt_df, time_increment=time_increment)
    nw = get_notewise_f_measure(df, gt_df)
    return (fw + nw) / 2


def get_framewise_f_measure(df, gt_df, time_increment=40):
    """
    Get the framewise F-measure of a dtaframe, given a ground truth dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe whose framewise F-measure to return.

    gt_df : pd.DataFrame
        The ground truth dataframe.

    time_increment : int
        The length of a frame to use when performing evaluations.

    Returns
    -------
    f_measure : float
        The framewise f_measure of the given dataframe.
    """
    gt_quant_df = gt_df.loc[:, ["pitch"]]
    gt_quant_df["onset"] = (gt_df["onset"] / time_increment).round().astype(int)
    gt_quant_df["offset"] = (
        ((gt_df["onset"] + gt_df["dur"]) / time_increment)
        .round()
        .astype(int)
        .clip(lower=gt_quant_df["onset"] + 1)
    )

    quant_df = df.loc[:, ["pitch"]]
    quant_df["onset"] = (df["onset"] / time_increment).round().astype(int)
    quant_df["offset"] = (
        ((df["onset"] + df["dur"]) / time_increment)
        .round()
        .astype(int)
        .clip(lower=quant_df["onset"] + 1)
    )

    # Create piano rolls
    length = int(max(1, quant_df["offset"].max(), gt_quant_df["offset"].max()))
    max_pitch = int(max(1, quant_df["pitch"].max(), gt_quant_df["pitch"].max()) + 1)
    pr = np.zeros((length, max_pitch))
    for _, note in quant_df.iterrows():
        pr[note.onset : note.offset, note.pitch] = 1

    gt_pr = np.zeros((length, max_pitch))
    for _, note in gt_quant_df.iterrows():
        gt_pr[note.onset : note.offset, note.pitch] = 1

    tp = np.sum(np.logical_and(gt_pr, pr))
    fp = np.sum(pr) - tp
    fn = np.sum(gt_pr) - tp

    _, _, f = get_f1(tp, fp, fn)
    return f


def get_notewise_f_measure(df, gt_df):
    """
    Get the notewise F-measure of a dataframe, given a ground truth dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe whose notewise F-measure to return.

    gt_df : pd.DataFrame
        The ground truth dataframe.

    time_increment : int
        The length of a frame to use for quantization when performing
        evaluations.

    Returns
    -------
    f_measure : float
        The notewise f_measure of the given dataframe, using mir_eval.
    """
    gt_pitches = np.array(gt_df["pitch"])
    gt_times = np.vstack((gt_df["onset"], gt_df["onset"] + gt_df["dur"])).T

    pitches = np.array(df["pitch"])
    times = np.vstack((df["onset"], df["onset"] + df["dur"])).T

    _, _, fm, _ = precision_recall_f1_overlap(
        gt_times,
        gt_pitches,
        times,
        pitches,
        onset_tolerance=50,
        pitch_tolerance=0,
        offset_ratio=None,
    )
    return fm


def get_f1(tp, fp, fn):
    """
    Get the precision, recall, and f1 of a given count of TPs, FPs, and FNs.

    Parameters
    ----------
    tp : int
        True positive count.

    fp : int
        False positive count.

    fn : int
        False negative count.

    Returns
    -------
    p : float
        Precision. 0 if tp == 0 (even if fp is also 0).

    r : float
        Recall. 0 if tp == 0 (even if fn is also 0).

    fm : float
        F1 score. 0 if tp == 0 (even if fn or fp are also 0).
    """
    if tp == 0:
        return 0, 0, 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    fm = 2 * p * r / (p + r)
    return p, r, fm
