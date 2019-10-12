"""Module containing methods to evaluate performance on Error Correction"""


def helpfulness(corrected, degraded, clean, timestep=40):
    """
    Get the helpfulness of an excerpt, given the degraded and clean versions.

    Parameters
    ----------
    corrected : pd.DataFrame
        The data frame of the corrected excerpt as output by a model.

    degraded : pd.DataFrame
        The degraded excerpt given as input to the model.

    clean : pd.DataFrame
        The clean excerpt expected as output.

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
    """
    raise NotImplementedError()