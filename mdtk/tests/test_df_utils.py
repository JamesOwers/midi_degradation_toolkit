import itertools

import pandas as pd

from mdtk.df_utils import NOTE_DF_SORT_ORDER, remove_pitch_overlaps, clean_df


CLEAN_INPUT_DF = pd.DataFrame({
    'track': list(range(9)) + [8],
    'dur': [50, 50, 40, 50, 50, 40, 50, 50, 40, 50],
    'onset': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
    'extra': 40,
    'pitch': [30, 20, 20, 30, 20, 20, 30, 20, 20, 20],
})

# Dict of [single_track][non_overlapping] -> result
CLEAN_RES_DFS = {
    False: {
        False: pd.DataFrame({
            'onset': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
            'track': list(range(9)) + [8],
            'pitch': [30, 20, 20, 30, 20, 20, 30, 20, 20, 20],
            'dur': [50, 50, 40, 50, 50, 40, 50, 50, 40, 50]
        }),
        True: pd.DataFrame({
            'onset': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
            'track': list(range(9)) + [8],
            'pitch': [30, 20, 20, 30, 20, 20, 30, 20, 20, 20],
            'dur': [50, 50, 40, 50, 50, 40, 50, 50, 1, 50]
        })
    },
    True: {
        False: pd.DataFrame({
            'onset': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
            'track': 0,
            'pitch': [20, 20, 30, 20, 20, 30, 20, 20, 30, 20],
            'dur': [40, 50, 50, 40, 50, 50, 40, 50, 50, 50]
        }),
        True: pd.DataFrame({
            'onset': [0, 0, 1, 1, 2, 2, 3],
            'track': 0,
            'pitch': [20, 30, 20, 30, 20, 30, 20],
            'dur': [1, 1, 1, 1, 1, 50, 50]
        })
    }
}


def test_clean_df():
    # Default, no arguments. multi-track, with overlaps
    # But correct sorting, columns, and index
    res = clean_df(CLEAN_INPUT_DF)
    assert res.equals(clean_df(CLEAN_INPUT_DF, single_track=False,
                               non_overlapping=False)), (
        "clean_df does not default to False, False for optional args."
    )

    for track, overlap in itertools.product([True, False], repeat=2):
        kwargs = {
            'single_track': track,
            'non_overlapping': overlap
        }
        res = clean_df(CLEAN_INPUT_DF, **kwargs)
        assert res.equals(CLEAN_RES_DFS[track][overlap]), (
            f"clean_df result incorrect for args: {kwargs}"
        )


def test_remove_pitch_overlaps():
    note_df_complex_overlap = pd.DataFrame({
        'onset': [0, 50, 75, 150, 200, 200, 300, 300, 300, 300],
        'track': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        'pitch': [10, 10, 10, 20, 10, 20, 30, 30, 10, 10],
        'dur': [50, 300, 25, 100, 125, 50, 50, 100, 100, 100]
    })
    note_df_complex_overlap_fixed = pd.DataFrame({
        'onset': [0, 50, 75, 150, 200, 200, 300, 300],
        'track': [0, 0, 0, 0, 0, 0, 0, 1],
        'pitch': [10, 10, 10, 20, 10, 20, 30, 10],
        'dur': [50, 25, 125, 50, 150, 50, 100, 100]
    })

    res = remove_pitch_overlaps(note_df_complex_overlap)
    assert note_df_complex_overlap_fixed.equals(res), (
        f"Complex overlap\n{note_df_complex_overlap}\nproduced\n{res}\n"
        f"instead of\n{note_df_complex_overlap_fixed}"
    )

test_clean_df()
