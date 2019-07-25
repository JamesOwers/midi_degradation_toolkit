import os
from glob import glob
import pandas as pd
import numpy as np
from mdtk.data_structures import Composition


DATA_LOC = 'data'


def test_working_dir():
    cwd = os.getcwd()
    assert (os.path.basename(cwd) == 'melody_gen' and
            os.path.exists(os.path.join(cwd, 'LICENSE'))), ("Please run tests "
            f"from the project root directory. Currently running from {cwd}")


def test_data_has_been_downloaded():
    print(os.path.dirname(os.path.realpath(__file__)))
    sizes = ['small', 'medium', 'large']
    parts = ['prime', 'cont_foil', 'cont_true']
    for size in sizes:
        for part in parts:
            path = os.path.join(DATA_LOC, 'raw',
                                f'PPDD-Jul2018_aud_mono_{size}', f'{part}_csv')
            assert os.path.isdir(f"{path}"), \
                (f"Could not find data at {path}. Please consult "
                 "README.md.")


def get_random_csv_filepath(seed=None, size=None, part=None):
    if seed:
        np.random.seed(seed)
    if size is None:
        size = np.random.choice(['small', 'medium', 'large'])
    if part is None:
        part = np.random.choice(['prime', 'cont_foil', 'cont_true'])
    csv_dir = os.path.join(DATA_LOC, 'raw', f'PPDD-Jul2018_aud_mono_{size}',
                           f'{part}_csv')
    return np.random.choice(glob(f"{csv_dir}/*.csv"))


class RandomFilepath:
    """Convenience class such that you can create a 'variable' which returns
    a random file path every time it is called e.g.
        random_path = RandomFilepath()
        random_path
            abc
        random_path
            efg
    """
    def __call__(self, seed=None, size=None, part=None):
        return get_random_csv_filepath(seed=seed, size=size, part=part)

    def __repr__(self):
        return get_random_csv_filepath()


random_path = RandomFilepath()


def get_random_compositions(nr_comps=1, **kwargs):
    if nr_comps == 1:
        return Composition(csv_path=str(random_path), **kwargs)
    return [Composition(csv_path=str(random_path), **kwargs)
            for ii in range(nr_comps)]


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


def test_all_composition_methods_and_attributes_with_random_files():
    compositions = [comp for comp in get_random_compositions(10)]
    for comp in compositions:
        comp.csv_path
        comp.note_df
        comp.quantization
        comp.quant_df
        comp.pianoroll
        comp.quanta_labels
        comp.plot()
        comp.synthesize()


def test_monophonic():
    compositions = [comp for comp in
                    get_random_compositions(10, monophonic=True)]
    for comp in compositions:
        assert np.sum(comp.pianoroll[:, :, 0]) == comp.pianoroll.shape[1]
