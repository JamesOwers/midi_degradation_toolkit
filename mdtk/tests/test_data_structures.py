import os
from glob import glob
import pandas as pd
import numpy as np
from mdtk.data_structures import (Composition, Pianoroll, read_note_csv,
                                  NOTE_DF_SORT_ORDER)


note_df_2pitch_aligned = pd.DataFrame(
    {'onset': [0, 0, 1, 2, 3, 4],
     'track' : 0,
     'pitch': [61, 60, 60, 60, 60, 60],
     'dur': [4, 1, 1, 0.5, 0.5, 2]}
)
note_df_2pitch_weird_times = pd.DataFrame(
    {'onset': [0, 0, 1.125, 1.370],
     'track' : 0,
     'pitch': [61, 60, 60, 60],
     'dur': [1.6, .9, .24, 0.125]}
)
note_df_with_silence = pd.DataFrame(
    {'onset': [0, 0, 1, 2, 3, 4],
     'track' : 0,
     'pitch': [61, 60, 60, 60, 60, 60],
     'dur': [3.75, 1, 1, 0.5, 0.5, 2]}
)
# midinote keyboard range from 0 to 127 inclusive
all_midinotes = list(range(0, 128))
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
all_pitch_df_tracks_overlaps = pd.DataFrame({
    'onset': [x for sublist in [np.arange(ii, len(all_midinotes)*2 + ii, 2)
                                for ii in range(nr_tracks)]
              for x in sublist],
    'pitch': all_midinotes * nr_tracks,
    'dur': [3] * (len(all_midinotes)*nr_tracks),
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

ALL_VALID_DF = [
    (note_df_2pitch_aligned.sort_values(NOTE_DF_SORT_ORDER)
                           .reset_index(drop=True)),
    (note_df_2pitch_weird_times.sort_values(NOTE_DF_SORT_ORDER)
                               .reset_index(drop=True)),
    (note_df_with_silence.sort_values(NOTE_DF_SORT_ORDER)
                         .reset_index(drop=True)),
    (all_pitch_df.sort_values(NOTE_DF_SORT_ORDER)
                 .reset_index(drop=True)),
    (all_pitch_df_tracks.sort_values(NOTE_DF_SORT_ORDER)
                        .reset_index(drop=True))
]


all_pitch_df.to_csv('./all_pitch_df.csv', index=False)
all_pitch_df_tracks.to_csv('./all_pitch_df_tracks.csv', index=False)
all_pitch_df_tracks_sparecol = all_pitch_df_tracks.copy(deep=True)
all_pitch_df_tracks_sparecol['sparecol'] = 'whoops'
all_pitch_df_tracks_sparecol.to_csv('./all_pitch_df_tracks_sparecol.csv',
                                    index=False)
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

ALL_CSVS = [
        './all_pitch_df.csv',
        './all_pitch_df_tracks.csv',
        './all_pitch_df_tracks_sparecol.csv',
        './all_pitch_df_tracks_sparecol_weirdorder.csv',
        './all_pitch_df_tracks_sparecol_noheader.csv'
    ]



# Function tests ==============================================================
def test_read_note_csv():
    df = read_note_csv('./all_pitch_df.csv', track=None)
    assert not df.equals(all_pitch_df)  # track names don't match
    df = read_note_csv('./all_pitch_df.csv', track='track')
    assert df.equals(all_pitch_df)
    df = read_note_csv('./all_pitch_df_tracks.csv', track='track', sort=False)
    assert df.equals(all_pitch_df_tracks[NOTE_DF_SORT_ORDER])
    df = read_note_csv('./all_pitch_df_tracks_sparecol.csv', track='track',
                       sort=False)
    comp_df = all_pitch_df_tracks_sparecol[NOTE_DF_SORT_ORDER + ['sparecol']]
    assert df.equals(comp_df.drop('sparecol', axis=1))
    df = read_note_csv('./all_pitch_df_tracks_sparecol.csv', track='track')
    assert not df.equals(comp_df.drop('sparecol', axis=1))  # sorting
    assert df.equals(
            (comp_df
                 .drop('sparecol', axis=1)
                 .sort_values(by=NOTE_DF_SORT_ORDER)
                 .reset_index(drop=True)
            )[NOTE_DF_SORT_ORDER])  # sorting
    df = read_note_csv('./all_pitch_df_tracks_sparecol_weirdorder.csv',
                       track='track', sort=False)
    assert df.equals(comp_df.drop('sparecol', axis=1))
    

    
 # Pianoroll class tests ======================================================
def test_pianoroll_all_pitches():
    pianoroll = Pianoroll(quant_df=all_pitch_df)
    assert (pianoroll == np.ones((1, 2, 128, 1), dtype='uint8')).all()
    

# TODO: test all note_on occur with sounding
    
# TODO: test all note_off occur with sounding

# TODO: test all sounding begin note_on and end_note_off



# Composition class tests =====================================================
# TODO: write import from csv tests    

def test_composition_df_assertions():
    assertion = False
    try:
        Composition(note_df=all_pitch_df_notrack)
    except AssertionError as e:
        if e.args == ("note_df must contain all columns in ['onset', 'track', "
                      "'pitch', 'dur']",):
            assertion = True
    assert assertion
    assertion = False
    
    assertion = False
    try:
        Composition(note_df=note_df_2pitch_aligned)
    except AssertionError as e:
        if e.args == ("note_df must be sorted by ['onset', 'track', 'pitch', "
                      "'dur'] and columns ordered",):
            assertion = True
    assert assertion
    assertion = False
    
    try:
        Composition(
            note_df=note_df_2pitch_aligned.sort_values(NOTE_DF_SORT_ORDER)
        )
    except AssertionError as e:
        if e.args == ("note_df must have a RangeIndex with integer steps",):
            assertion = True
    assert assertion
    
    assert Composition(
            note_df=(
                note_df_2pitch_aligned
                    .sort_values(NOTE_DF_SORT_ORDER)
                    .reset_index(drop=True)
            )
        )


def test_composition_all_pitches():
    c = Composition(note_df=all_pitch_df, quantization=1)
    pr = np.ones_like(c.pianoroll)
    assert (pr == c.pianoroll).all()



# TODO: reimplement this if and when we implement auto fix of note_df
#def test_auto_sort_onset_and_pitch():
#    comp = Composition(note_df=note_df_2pitch_aligned, fix_note_df=True)
#    assert comp.note_df.equals(
#        note_df_2pitch_aligned
#            .sort_values(['onset', 'pitch'])
#            .reset_index(drop=True)
#    )


def test_not_ending_in_silence():
    for df in ALL_VALID_DF:
        comp = Composition(note_df=df)
        assert not (comp.pianoroll[:, 0, :, -1] == 0).all(), (f'{df}',
                   f'{comp.plot()} {comp.pianoroll}')


def test_nr_note_on_equals_nr_notes():
    for df in ALL_VALID_DF:
        comp = Composition(note_df=df)
        pianoroll = comp.pianoroll
        assert np.sum(pianoroll[:, 1, :, :]) == comp.note_df.shape[0]


def test_all_composition_methods_and_attributes():
    compositions = [Composition(comp)
                    for comp in ALL_VALID_DF]
    for comp in compositions:
        comp.csv_path
        comp.note_df
        comp.quantization
        comp.quant_df
        comp.pianoroll
        comp.quanta_labels
        comp.plot()
        comp.synthesize()

# TODO: Check if anything alters input data - loop over all functions and
#       methods


# Cleanup =====================================================================
# TODO: This isn't technichally a test...should probably be some other function
#       look up the proper way to do this.
def test_remove_csvs(): 
    for csv in ALL_CSVS:
        os.remove(csv)