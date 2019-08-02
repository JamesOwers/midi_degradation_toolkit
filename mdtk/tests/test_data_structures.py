import os
from glob import glob
import pandas as pd
import numpy as np
from mdtk.data_structures import Composition


note_df_2pitch_aligned = pd.DataFrame(
    {'onset': [0, 0, 1, 2, 3, 4],
     'pitch': [61, 60, 60, 60, 60, 60],
     'dur': [4, 1, 1, 0.5, 0.5, 2]}
)
note_df_2pitch_weird_times = pd.DataFrame(
    {'onset': [0, 0, 1.125, 1.370],
     'pitch': [61, 60, 60, 60],
     'dur': [1.6, .9, .24, 0.125]}
)
note_df_with_silence = pd.DataFrame(
    {'onset': [0, 0, 1, 2, 3, 4],
     'pitch': [61, 60, 60, 60, 60, 60],
     'dur': [3.75, 1, 1, 0.5, 0.5, 2]}
)
# midinote keyboard range from 0 to 127 inclusive
all_midinotes = list(range(0, 128))
all_pitch_df = pd.DataFrame({
    'onset': [0] * len(all_midinotes),
    'pitch': all_midinotes,
    'dur': [1] * len(all_midinotes)
})

nr_tracks = 3
track_names = np.random.choice(np.arange(10), replace=False, size=nr_tracks)
all_pitch_df_tracks = pd.DataFrame({
    'onset': [x for sublist in [np.arange(ii, len(all_midinotes)*2 + ii, 2)
                                for ii in range(nr_tracks)]
              for x in sublist],
    'pitch': all_midinotes * nr_tracks,
    'dur': [2] * (len(all_midinotes)*nr_tracks),
    'track': [x for sublist in [[name]*len(all_midinotes)
                                for name in track_names]
              for x in sublist]
})

ALL_DF = [
    note_df_2pitch_aligned,
    note_df_2pitch_weird_times,
    note_df_with_silence,
    all_pitch_df
]


def test_assert_sort_onset_and_pitch():
    assertion = False
    try:
        Composition(note_df=note_df_2pitch_aligned)
    except AssertionError:
        assertion = True
    assert assertion


def test_all_pitches():
    c = Composition(note_df=all_pitch_df, sort_note_df=False,
                    check_sorted=True, quantization=1)
    pr = np.ones_like(c.pianoroll)
    assert (pr[:-1] == c.pianoroll[:-1]).all()  # don't consider silence token


def test_auto_sort_onset_and_pitch():
    comp = Composition(note_df=note_df_2pitch_aligned, sort_note_df=True)
    assert comp.note_df.equals(
        note_df_2pitch_aligned
            .sort_values(['onset', 'pitch'])
            .reset_index(drop=True)
    )


def test_not_ending_in_silence():
    for df in ALL_DF:
        comp = Composition(note_df=df, sort_note_df=True)
        assert comp.pianoroll[-1, -1, 0] != 1


def test_nr_note_offs_equals_nr_notes():
    for df in ALL_DF:
        comp = Composition(note_df=df, sort_note_df=True)
        pianoroll = comp.pianoroll
        assert np.sum(pianoroll[:-1, :, 1]) == comp.note_df.shape[0]


def test_all_composition_methods_and_attributes():
    compositions = [Composition(comp, sort_note_df=True) for comp in ALL_DF]
    for comp in compositions:
        comp.csv_path
        comp.note_df
        comp.quantization
        comp.quant_df
        comp.pianoroll
        comp.quanta_labels
        comp.plot()
        comp.synthesize()



