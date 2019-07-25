import numpy as np
import os
from glob import glob

from mdtk.downloaders import PPDDSept2018Monophonic
from mdtk.data_structures import Composition


DOWNLOADERS = [PPDDSept2018Monophonic]


#def test_working_dir():
#    cwd = os.getcwd()
#    assert (os.path.basename(cwd) == 'melody_gen' and
#            os.path.exists(os.path.join(cwd, 'LICENSE'))), ("Please run tests "
#            f"from the project root directory. Currently running from {cwd}")


#def get_random_csv_filepath(seed=None, size=None, part=None):
#    if seed:
#        np.random.seed(seed)
#    if size is None:
#        size = np.random.choice(['small', 'medium', 'large'])
#    if part is None:
#        part = np.random.choice(['prime', 'cont_foil', 'cont_true'])
#    csv_dir = os.path.join(DATA_LOC, 'raw', f'PPDD-Jul2018_aud_mono_{size}',
#                           f'{part}_csv')
#    return np.random.choice(glob(f"{csv_dir}/*.csv"))
#
#
#class RandomFilepath:
#    """Convenience class such that you can create a 'variable' which returns
#    a random file path every time it is called e.g.
#        random_path = RandomFilepath()
#        random_path
#            abc
#        random_path
#            efg
#    """
#    def __call__(self, seed=None, size=None, part=None):
#        return get_random_csv_filepath(seed=seed, size=size, part=part)
#
#    def __repr__(self):
#        return get_random_csv_filepath()


#random_path = RandomFilepath()


#def get_random_compositions(nr_comps=1, **kwargs):
#    if nr_comps == 1:
#        return Composition(csv_path=str(random_path), **kwargs)
#    return [Composition(csv_path=str(random_path), **kwargs)
#            for ii in range(nr_comps)]


#def test_data_has_been_downloaded():
#    print(os.path.dirname(os.path.realpath(__file__)))
#    sizes = ['small', 'medium', 'large']
#    parts = ['prime', 'cont_foil', 'cont_true']
#    for size in sizes:
#        for part in parts:
#            path = os.path.join(DATA_LOC, 'raw',
#                                f'PPDD-Jul2018_aud_mono_{size}', f'{part}_csv')
#            assert os.path.isdir(f"{path}"), \
#                (f"Could not find data at {path}. Please consult "
#                 "README.md.")

#def test_monophonic():
#    compositions = [comp for comp in
#                    get_random_compositions(10, monophonic=True)]
#    for comp in compositions:
#        assert np.sum(comp.pianoroll[:, :, 0]) == comp.pianoroll.shape[1]


#def test_all_composition_methods_and_attributes_with_random_files():
#    compositions = [comp for comp in get_random_compositions(10)]
#    for comp in compositions:
#        comp.csv_path
#        comp.note_df
#        comp.quantization
#        comp.quant_df
#        comp.pianoroll
#        comp.quanta_labels
#        comp.plot()
#        comp.synthesize()