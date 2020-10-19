import itertools

import pandas as pd

from mdtk.df_utils import clean_df, get_random_excerpt, remove_pitch_overlaps

CLEAN_INPUT_DF = pd.DataFrame(
    {
        "track": list(range(9)) + [8],
        "dur": [50, 50, 40, 50, 50, 40, 50, 50, 40, 50],
        "onset": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
        "extra": 40,
        "pitch": [30, 20, 20, 30, 20, 20, 30, 20, 20, 20],
        "velocity": list(range(9)) + [8],
    }
)

# Dict of [single_track][non_overlapping] -> result
CLEAN_RES_DFS = {
    False: {
        False: pd.DataFrame(
            {
                "onset": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
                "track": list(range(9)) + [8],
                "pitch": [30, 20, 20, 30, 20, 20, 30, 20, 20, 20],
                "dur": [50, 50, 40, 50, 50, 40, 50, 50, 40, 50],
                "velocity": list(range(9)) + [8],
            }
        ),
        True: pd.DataFrame(
            {
                "onset": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
                "track": list(range(9)) + [8],
                "pitch": [30, 20, 20, 30, 20, 20, 30, 20, 20, 20],
                "dur": [50, 50, 40, 50, 50, 40, 50, 50, 1, 50],
                "velocity": list(range(9)) + [8],
            }
        ),
    },
    True: {
        False: pd.DataFrame(
            {
                "onset": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
                "track": 0,
                "pitch": [20, 20, 30, 20, 20, 30, 20, 20, 30, 20],
                "dur": [40, 50, 50, 40, 50, 50, 40, 50, 50, 50],
                "velocity": [2, 1, 0, 5, 4, 3, 8, 7, 6, 8],
            }
        ),
        True: pd.DataFrame(
            {
                "onset": [0, 0, 1, 1, 2, 2, 3],
                "track": 0,
                "pitch": [20, 30, 20, 30, 20, 30, 20],
                "dur": [1, 1, 1, 1, 1, 50, 50],
                "velocity": [1, 0, 4, 3, 7, 6, 8],
            }
        ),
    },
}


def test_clean_df():
    # Default, no arguments. multi-track, with overlaps
    # But correct sorting, columns, and index
    res = clean_df(CLEAN_INPUT_DF)
    assert res.equals(
        clean_df(CLEAN_INPUT_DF, single_track=False, non_overlapping=False)
    ), "clean_df does not default to False, False for optional args."

    for track, overlap in itertools.product([True, False], repeat=2):
        kwargs = {"single_track": track, "non_overlapping": overlap}
        prior = CLEAN_INPUT_DF.copy()
        res = clean_df(CLEAN_INPUT_DF, **kwargs)
        assert CLEAN_INPUT_DF.equals(prior), "clean_df changed input df"
        assert res.equals(
            CLEAN_RES_DFS[track][overlap]
        ), f"clean_df result incorrect for args: {kwargs}"


def test_remove_pitch_overlaps():
    note_df_complex_overlap = pd.DataFrame(
        {
            "onset": [0, 50, 75, 150, 200, 200, 300, 300, 300, 300],
            "track": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            "pitch": [10, 10, 10, 20, 10, 20, 30, 30, 10, 10],
            "dur": [50, 300, 25, 100, 125, 50, 50, 100, 100, 100],
            "velocity": list(range(10)),
        }
    )
    note_df_complex_overlap_fixed = pd.DataFrame(
        {
            "onset": [0, 50, 75, 150, 200, 200, 300, 300],
            "track": [0, 0, 0, 0, 0, 0, 0, 1],
            "pitch": [10, 10, 10, 20, 10, 20, 30, 10],
            "dur": [50, 25, 125, 50, 150, 50, 100, 100],
            "velocity": [0, 1, 2, 3, 4, 5, 7, 9],
        }
    )

    prior = note_df_complex_overlap.copy()
    res = remove_pitch_overlaps(note_df_complex_overlap)
    assert prior.equals(note_df_complex_overlap), (
        "remove_pitch_overlaps " "changed input df"
    )
    assert note_df_complex_overlap_fixed.equals(res), (
        f"Complex overlap\n{note_df_complex_overlap}\nproduced\n{res}\n"
        f"instead of\n{note_df_complex_overlap_fixed}"
    )

    short_df = pd.DataFrame({"onset": [0], "track": 0, "pitch": 0, "dur": 0})
    assert short_df.equals(remove_pitch_overlaps(short_df))


def test_get_random_excerpt():
    NUM_NOTES = 50
    NOTE_DURATION = 50
    note_df = pd.DataFrame(
        {
            "onset": [NOTE_DURATION * i for i in range(NUM_NOTES)],
            "track": 0,
            "pitch": list(range(NUM_NOTES)),
            "dur": NOTE_DURATION,
        }
    )

    # Normal usage
    prior = note_df.copy()
    for _ in range(50):
        min_notes = 10
        excerpt_length = (NUM_NOTES - 1) * NOTE_DURATION + 1  # All notes fit
        start = 500
        end = 600
        excerpt = get_random_excerpt(
            note_df,
            min_notes=min_notes,
            excerpt_length=excerpt_length,
            first_onset_range=(start, end),
            iterations=1,
        )
        assert prior.equals(note_df), "get_random_excerpt changed input df"
        assert len(excerpt) >= min_notes, "Too few notes in excerpt"
        assert excerpt.iloc[-1].pitch == NUM_NOTES - 1, (
            "Full excerpt_length " "not returned"
        )
        excerpt_start = excerpt.iloc[0].onset
        assert start <= excerpt_start < end
        assert all(
            excerpt["onset"].to_numpy()
            == [excerpt_start + NOTE_DURATION * i for i in range(len(excerpt))]
        ), "All note onsets did not shift correctly"

    # Not enough iterations
    assert (
        get_random_excerpt(
            note_df,
            min_notes=min_notes,
            excerpt_length=excerpt_length,
            first_onset_range=(start, end),
            iterations=0,
        )
        is None
    ), "Did not return None with iterations=0"
    assert prior.equals(note_df), "get_random_excerpt changed input df"

    # Too large min_notes
    assert (
        get_random_excerpt(
            note_df,
            min_notes=NUM_NOTES + 1,
            excerpt_length=excerpt_length,
            first_onset_range=(start, end),
            iterations=100,
        )
        is None
    ), "Did not return None with min_notes too large"
    assert prior.equals(note_df), "get_random_excerpt changed input df"

    # Too short excerpt_length
    excerpt_length = NOTE_DURATION * (min_notes - 1) - 1
    assert (
        get_random_excerpt(
            note_df,
            min_notes=NUM_NOTES + 1,
            excerpt_length=excerpt_length,
            first_onset_range=(start, end),
            iterations=100,
        )
        is None
    ), "Did not return None with excerpt_length too short"
    assert prior.equals(note_df), "get_random_excerpt changed input df"
