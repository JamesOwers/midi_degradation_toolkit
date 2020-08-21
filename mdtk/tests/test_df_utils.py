import os
import pandas as pd
import numpy as np
import pytest
import re
from mdtk.df_utils import NOTE_DF_SORT_ORDER, remove_overlaps, clean_df


# Test note DataFrames ========================================================
# Two notes in the same track overlap and have the same pitch
note_df_overlapping_pitch = pd.DataFrame({
    'onset': [0, 1],
    'track' : 0,
    'pitch': [60, 60],
    'dur': [2, 1]
})
note_df_overlapping_pitch_fix = pd.DataFrame({
    'onset': [0, 1],
    'track' : 0,
    'pitch': [60, 60],
    'dur': [1, 1]
})
# Two notes in the same track overlap but have different pitches
note_df_overlapping_note = pd.DataFrame({
    'onset': [0, 1],
    'track' : 0,
    'pitch': [60, 61],
    'dur': [2, 1]
})
# Incorrect pitch sort order, polyphonic track 0
note_df_2pitch_aligned = pd.DataFrame({
    'onset': [0, 0, 1, 2, 3, 4],
    'track' : 0,
    'pitch': [61, 60, 60, 60, 60, 60],
    'dur': [4, 1, 1, 0.5, 0.5, 2]
})
note_df_ms = pd.DataFrame({
    'onset': [0, 0, 40, 80, 120, 160],
    'track' : 0,
    'pitch': [61, 60, 60, 60, 60, 60],
    'dur': [160, 40, 40, 20, 20, 80]
})
note_df_ms_quant = pd.DataFrame({
    'onset': [0, 0, 1, 2, 3, 4],
    'track' : 0,
    'pitch': [61, 60, 60, 60, 60, 60],
    'dur': [4, 1, 1, 1, 1, 2]  # N.B. 20*1/40 = 0.5 but this rounds down to 0
                                # the quant function adjusts this to 1
})
# Not quantized, polyphonic
note_df_2pitch_weird_times = pd.DataFrame({
    'onset': [0, 0, 1.125, 1.370],
    'track' : 0,
    'pitch': [60, 61, 60, 60],
    'dur': [.9, 1.6, .24, 0.125]
})
# Expected quantization of the above at 12 divisions per integer
# N.B. Although the duration of .24 would quantize to 3, this would create
# an overlapping pitch, so it must be reduced to 2
note_df_2pitch_weird_times_quant = pd.DataFrame({
    'onset': [0, 0, 14, 16],
    'track' : 0,
    'pitch': [60, 61, 60, 60],
    'dur': [11, 19, 2, 2]
})
# silence between 3.75 and 4
note_df_with_silence = pd.DataFrame({
    'onset': [0, 0, 1, 2, 3, 4],
    'track' :[0, 1, 0, 0, 0, 0],
    'pitch': [60, 61, 60, 60, 60, 60],
    'dur': [1, 3.75, 1, 0.5, 0.5, 2]
})
note_df_odd_names = pd.DataFrame({
    'note_on': [0, 0, 1, 2, 3, 4],
    'midinote': [60, 61, 60, 60, 60, 60],
    'duration': [1, 3.75, 1, 0.5, 0.5, 2],
    'ch' :[0, 1, 0, 0, 0, 0]
})
note_df_complex_overlap = pd.DataFrame({
    'onset': [0, 50, 75, 150, 200, 200, 300, 300, 300],
    'track': [0, 0, 0, 0, 0, 0, 0, 0, 1],
    'pitch': [10, 10, 10, 20, 10, 20, 30, 30, 10],
    'dur': [50, 300, 25, 100, 125, 50, 50, 100, 100]
})
note_df_complex_overlap_fixed = pd.DataFrame({
    'onset': [0, 50, 75, 150, 200, 200, 300, 300],
    'track': [0, 0, 0, 0, 0, 0, 0, 1],
    'pitch': [10, 10, 10, 20, 10, 20, 30, 10],
    'dur': [50, 25, 125, 50, 150, 50, 100, 100]
})
# midinote keyboard range from 0 to 127 inclusive
all_midinotes = list(range(0, 128))
# All pitches played simultaneously for 1 quantum
all_pitch_df_notrack = pd.DataFrame({
    'onset': [0] * len(all_midinotes),
    'pitch': all_midinotes,
    'dur': [1] * len(all_midinotes)
})
all_pitch_df = all_pitch_df_notrack.copy()
all_pitch_df['track'] = 1
all_pitch_df_wrongorder = all_pitch_df.copy()
all_pitch_df = all_pitch_df[NOTE_DF_SORT_ORDER]

nr_tracks = 3
track_names = np.random.choice(np.arange(10), replace=False, size=nr_tracks)
track_names.sort()
# All pitches played for duration 2 offset by 1 each, each track starting
# one onset after one another
all_pitch_df_tracks = pd.DataFrame({
    'onset': [x for sublist in [np.arange(ii, len(all_midinotes)*2 + ii, 2)
                                for ii in range(nr_tracks)]
              for x in sublist],
    'track': [x for sublist in [[name]*len(all_midinotes)
                                for name in track_names]
              for x in sublist],
    'pitch': all_midinotes * nr_tracks,
    'dur': [2] * (len(all_midinotes)*nr_tracks)
})
all_pitch_df_tracks_overlaps = pd.DataFrame({
    'onset': [x for sublist in [np.arange(ii, len(all_midinotes)*2 + ii, 2)
                                for ii in range(nr_tracks)]
              for x in sublist],
    'track': [x for sublist in [[name]*len(all_midinotes)
                                for name in track_names]
              for x in sublist],
    'pitch': all_midinotes * nr_tracks,
    'dur': [3] * (len(all_midinotes)*nr_tracks)
})
all_pitch_df_tracks_sparecol = all_pitch_df_tracks.copy(deep=True)
all_pitch_df_tracks_sparecol['sparecol'] = 'whoops'
df_all_mono = pd.DataFrame({
    'onset': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    'track' : [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    'pitch': [60, 61, 62, 63, 70, 71, 72, 73, 80, 81, 82, 83],
    'dur': [1]*(4*3)
}).sort_values(by=NOTE_DF_SORT_ORDER).reset_index(drop=True)
df_some_mono = pd.DataFrame({
    'onset': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    'track' : [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    'pitch': [60, 61, 62, 63, 70, 71, 72, 73, 80, 81, 82, 83],
    'dur': [1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1]
}).sort_values(by=NOTE_DF_SORT_ORDER).reset_index(drop=True)
df_all_poly = pd.DataFrame({
    'onset': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    'track' : [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    'pitch': [60, 61, 62, 63, 70, 71, 72, 73, 80, 81, 82, 83],
    'dur': [2, 1, 1, 1, 4, 1, 1, 1, 3, 1, 1, 1]
}).sort_values(by=NOTE_DF_SORT_ORDER).reset_index(drop=True)
note_df_with_silence = pd.DataFrame({
    'onset': [0, 0, 1, 2, 3, 4],
    'track' :[0, 1, 0, 0, 0, 0],
    'pitch': [60, 61, 60, 60, 60, 60],
    'dur': [1, 3.75, 1, 0.5, 0.5, 2]
})

ALL_DF = {
    'note_df_overlapping_pitch': note_df_overlapping_pitch,
    'note_df_overlapping_note': note_df_overlapping_note,
    'note_df_2pitch_aligned': note_df_2pitch_aligned,
    'note_df_odd_names': note_df_odd_names,
    'note_df_2pitch_weird_times': note_df_2pitch_weird_times,
    'note_df_2pitch_weird_times_quant': note_df_2pitch_weird_times_quant,
    'note_df_with_silence': note_df_with_silence,
    'all_pitch_df_notrack': all_pitch_df_notrack,
    'all_pitch_df_wrongorder': all_pitch_df_wrongorder,
    'all_pitch_df': all_pitch_df,
    'all_pitch_df_tracks': all_pitch_df_tracks,
    'all_pitch_df_tracks_overlaps': all_pitch_df_tracks_overlaps,
    'all_pitch_df_tracks_sparecol': all_pitch_df_tracks_sparecol,
    'df_all_mono': df_all_mono,
    'df_some_mono': df_some_mono,
    'df_all_poly': df_all_poly
}

sort_err = (f"note_df must be sorted by {NOTE_DF_SORT_ORDER} and columns "
            "ordered")
missing_col_err = (f"note_df must contain all columns in {NOTE_DF_SORT_ORDER}")
col_order_err = (f"note_df colums must be in order: {NOTE_DF_SORT_ORDER}")
extra_col_err = (f"note_df must only contain columns in {NOTE_DF_SORT_ORDER}")

ASSERTION_ERRORS = {
    'note_df_overlapping_pitch': None,
    'note_df_overlapping_note': None,
    'note_df_2pitch_aligned': sort_err,
    'note_df_ms': sort_err,
    'note_df_2pitch_weird_times': None,
    'note_df_2pitch_weird_times_quant': None,
    'note_df_with_silence': None,
    'note_df_odd_names': missing_col_err,
    'all_pitch_df_notrack': missing_col_err,
    'all_pitch_df_wrongorder': col_order_err,
    'all_pitch_df': None,
    'all_pitch_df_tracks': sort_err,
    'all_pitch_df_tracks_overlaps': sort_err,
    'all_pitch_df_tracks_sparecol': extra_col_err,
    'df_all_mono': None,
    'df_some_mono': None,
    'df_all_poly': None
}

overlap_warn = ('WARNING: Track(s) {bad_tracks} has an overlapping note '
                'at pitch(es) {bad_pitches}. This can lead to '
                'unexpected results.')

WARNINGS = {
    'note_df_overlapping_pitch': overlap_warn.format(bad_tracks=[0],
                                                     bad_pitches=[60])
}

def fix_sort(df):
    return (
        df
            .sort_values(by=NOTE_DF_SORT_ORDER)
            .reset_index(drop=True)
    )[NOTE_DF_SORT_ORDER]

ALL_VALID_DF = {
    'note_df_overlapping_pitch': note_df_overlapping_pitch_fix,
    'note_df_overlapping_note': note_df_overlapping_note,
    'note_df_2pitch_aligned': fix_sort(note_df_2pitch_aligned),
    'note_df_ms': fix_sort(note_df_ms),
    'note_df_2pitch_weird_times': note_df_2pitch_weird_times,
    'note_df_2pitch_weird_times_quant': note_df_2pitch_weird_times_quant,
    'note_df_with_silence': note_df_with_silence,
    'note_df_odd_names': note_df_with_silence,
    'all_pitch_df_notrack': all_pitch_df,
    'all_pitch_df_wrongorder': all_pitch_df,
    'all_pitch_df': all_pitch_df,
    'all_pitch_df_tracks': fix_sort(all_pitch_df_tracks),
    'all_pitch_df_tracks_overlaps': fix_sort(all_pitch_df_tracks_overlaps),
    'all_pitch_df_tracks_sparecol': fix_sort(all_pitch_df_tracks_sparecol),
    'df_all_mono': df_all_mono,
    'df_some_mono': df_some_mono,
    'df_all_poly': df_all_poly
}


for name, df in ALL_DF.items():
    df.to_csv(f'./{name}.csv', index=False)
note_df_complex_overlap.to_csv('./note_df_complex_overlap.csv', index=False)

all_pitch_df_tracks_sparecol.to_csv(
        './all_pitch_df_tracks_sparecol_noheader.csv',
        index=False,
        header=False
    )
weird_col_order = ['pitch','track','sparecol','dur','onset']
all_pitch_df_tracks_sparecol[weird_col_order].to_csv(
        './all_pitch_df_tracks_sparecol_weirdorder.csv',
        index=False
    )

ALL_CSV = [f'./{name}.csv' for name in ALL_DF.keys()]
ALL_CSV += ['./all_pitch_df_tracks_sparecol_noheader.csv',
            './all_pitch_df_tracks_sparecol_weirdorder.csv',
            './note_df_complex_overlap.csv']

default_read_note_csv_kwargs = dict(
    onset='onset',
    pitch='pitch',
    dur='dur',
    track='track',
    sort=True,
    header='infer'
)

ALL_CSV_KWARGS = {
    './note_df_overlapping_pitch.csv': default_read_note_csv_kwargs,
    './note_df_overlapping_note.csv': default_read_note_csv_kwargs,
    './note_df_2pitch_aligned.csv': default_read_note_csv_kwargs,
    './note_df_odd_names.csv': dict(
        onset='note_on',
        pitch='midinote',
        dur='duration',
        track='ch',
        sort=True,
        header='infer'
    ),
    './note_df_2pitch_weird_times.csv': default_read_note_csv_kwargs,
    './note_df_2pitch_weird_times_quant.csv': default_read_note_csv_kwargs,
    './note_df_with_silence.csv': default_read_note_csv_kwargs,
    './all_pitch_df_notrack.csv': dict(
        onset='onset',
        pitch='pitch',
        dur='dur',
        track=None,
        sort=True,
        header='infer'
    ),
    './all_pitch_df_wrongorder.csv': default_read_note_csv_kwargs,
    './all_pitch_df.csv': default_read_note_csv_kwargs,
    './all_pitch_df_tracks.csv': default_read_note_csv_kwargs,
    './all_pitch_df_tracks_overlaps.csv': default_read_note_csv_kwargs,
    './all_pitch_df_tracks_sparecol.csv': default_read_note_csv_kwargs,
    './df_all_mono.csv': default_read_note_csv_kwargs,
    './df_some_mono.csv': default_read_note_csv_kwargs,
    './df_all_poly.csv': default_read_note_csv_kwargs,
    './all_pitch_df_tracks_sparecol_noheader.csv':
        dict(
            onset=0,
            pitch=2,
            dur=3,
            track=1,
            sort=True,
            header=None
        ),
    './all_pitch_df_tracks_sparecol_weirdorder.csv':
        default_read_note_csv_kwargs,
    './note_df_complex_overlap.csv':
        default_read_note_csv_kwargs
}

# Function tests ==============================================================
def test_read_note_csv():
    df = read_note_csv('./all_pitch_df.csv', track=None)
    assert not df.equals(all_pitch_df)  # track names don't match
    all_pitch_df_track_name_change = all_pitch_df.copy()
    all_pitch_df_track_name_change['track'] = 0
    assert df.equals(all_pitch_df_track_name_change)
    df = read_note_csv('./all_pitch_df.csv', track='track')
    assert df.equals(all_pitch_df)
    df = read_note_csv('./all_pitch_df_tracks.csv', sort=False)
    assert df.equals(all_pitch_df_tracks[NOTE_DF_SORT_ORDER]), (
            f"{df}\n\n\n{all_pitch_df_tracks[NOTE_DF_SORT_ORDER]}")
    df = read_note_csv('./all_pitch_df_tracks_sparecol.csv', sort=False)
    comp_df = all_pitch_df_tracks_sparecol[NOTE_DF_SORT_ORDER + ['sparecol']]
    assert df.equals(comp_df.drop('sparecol', axis=1))
    df = read_note_csv('./all_pitch_df_tracks_sparecol.csv')
    assert not df.equals(comp_df.drop('sparecol', axis=1))  # sorting
    assert df.equals(
            (comp_df
                 .drop('sparecol', axis=1)
                 .sort_values(by=NOTE_DF_SORT_ORDER)
                 .reset_index(drop=True)
            )[NOTE_DF_SORT_ORDER])  # sorting
    df = read_note_csv('./all_pitch_df_tracks_sparecol_weirdorder.csv',
                       sort=False)
    assert df.equals(comp_df.drop('sparecol', axis=1))
    assert (
        read_note_csv('./note_df_with_silence.csv').equals(
            read_note_csv('./note_df_odd_names.csv', onset='note_on',
                          track='ch', pitch='midinote', dur='duration'))
        )
    assert all(read_note_csv('./note_df_complex_overlap.csv',
                             flatten_tracks=True)['track'] == 0), (
        "flatten_tracks=True didn't set all tracks to 0."
    )


def test_remove_overlaps():
    res = remove_overlaps(note_df_complex_overlap)
    assert note_df_complex_overlap_fixed.equals(res), (
        f"Complex overlap\n{note_df_complex_overlap}\nproduced\n{res}\n"
        f"instead of\n{note_df_complex_overlap_fixed}"
    )





# Cleanup =====================================================================
def test_remove_csvs():
    for csv in ALL_CSV:
        os.remove(csv)
