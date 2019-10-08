"""classes to use in conjunction with pytorch dataloaders"""
import pandas as pd
import numpy as np

def df_to_one_hots(df, min_pitch=0, max_pitch=127, frame_length=40,
                   max_shift=1000):
    """
    Convert a given pandas DataFrame into a sequence of one-hot vectors
    representing commands (note_on, note_off, shift, and EOF).

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame which we will convert into one-hots.

    min_pitch : int
        The minimum pitch at which notes will occur.

    max_pitch : int
        The maximum pitch at which notes will occur.

    frame_length : int
        The length of a single frame, in milliseconds.

    max_shift : int
        The maximum shift length, in milliseconds. Must be divisible by
        frame_length.

    Returns
    -------
    one_hot : np.ndarray
        An array containing the one-hot vectors describing the given DataFrame.
    """
    # Input validation
    assert max_shift % frame_length == 0, ("max_shift must be divisible by "
                                           "frame_length.")
    assert max_pitch >= min_pitch, "max_pitch must be >= min_pitch."
    assert frame_length > 0, "frame_length must be positive."
    assert max_shift > 0, "max_shift must be positive"

    num_pitches = max_pitch - min_pitch + 1
    num_shifts = max_shift / frame_length
    # note_on, note_off, shift, EOF
    vector_length = num_pitches * 2 + num_shifts + 1

    ONSET = 0
    OFFSET = 1

    # Get all note on and offs
    note_actions = []
    offsets = df['onset'] + df['dur']
    for note, offset in zip(df.iterrows(), offsets):
        note_actions.append((convert_time(note['onset'], frame_length),
                             ONSET, note['pitch']))
        note_actions.append((convert_time(offset, frame_length),
                             OFFSET, note['pitch']))

    # Sort the note actions by time, then on/off, then pitch
    note_actions = sorted(note_actions)

    # Create one-hot sequence
    one_hots = []
    current_frame = note_actions[0][0]
    for frame, action, pitch in note_actions:
        # Shift
        # TODO: Do we want to support iterative shifts like this or just error
        while frame != current_frame:
            diff = min(num_shifts, frame - current_frame)
            shift_vector = np.zeros(vector_length)
            shift_vector[2 * num_pitches + diff] = 1
            one_hots.append(shift_vector)
            current_frame += diff

        # Perform action
        assert min_pitch <= pitch <= max_pitch, (
            f"All pitches must be in range [{min_pitch}, {max_pitch}]."
        )
        action_vector = np.zeros(vector_length)
        index = pitch if action == ONSET else pitch + num_pitches
        action_vector[index] = 1
        one_hots.append(action_vector)

    # Add EOF
    eof = np.zeros(vector_length)
    eof[-1] = 1
    one_hots.append(eof)

    return np.array(one_hots)
