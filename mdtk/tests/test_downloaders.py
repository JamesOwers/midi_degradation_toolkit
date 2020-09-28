import os
import shutil
import urllib

from mdtk.downloaders import (
    PianoMidi,
    PPDDSep2018Monophonic,
    PPDDSep2018Polyphonic,
    make_directory,
)

DOWNLOADERS = [PPDDSep2018Monophonic, PPDDSep2018Polyphonic, PianoMidi]
USER_HOME = os.path.expanduser("~")
TEST_CACHE_PATH = os.path.join(USER_HOME, ".mdtk_test_cache")


def test_make_directory():
    make_directory(TEST_CACHE_PATH, overwrite=True)


def test_links_exist():
    for Downloader in DOWNLOADERS:
        if Downloader is PianoMidi:
            # Invalid urls for piano-midi always return code 200.
            # Can't test this way.
            continue
        downloader = Downloader(cache_path=TEST_CACHE_PATH)
        for url in downloader.download_urls:
            with urllib.request.urlopen(url) as response:
                code = response.getcode()
            assert code == 200


def test_PPDDSep2018Monophonic_download_midi():
    _ = PPDDSep2018Monophonic(cache_path=TEST_CACHE_PATH, sizes=["small", "medium"])
    # TODO: test downloading another way, this is too slow
    # output_path = os.path.join(TEST_CACHE_PATH, downloader.dataset_name, "midi")
    # downloader.download_midi(output_path)
    # assert len(os.listdir(output_path)) == 1100


def test_PPDDSep2018Polyphonic_download_midi():
    _ = PPDDSep2018Polyphonic(cache_path=TEST_CACHE_PATH, sizes=["small", "medium"])
    # TODO: test downloading another way, this is too slow
    # output_path = os.path.join(TEST_CACHE_PATH, downloader.dataset_name, "midi")
    # downloader.download_midi(output_path)
    # assert len(os.listdir(output_path)) == 1100


def test_PianoMidi_download_midi():
    _ = PianoMidi(cache_path=TEST_CACHE_PATH)
    # TODO: test downloading another way, this is too slow
    # output_path = os.path.join(TEST_CACHE_PATH, downloader.dataset_name, "midi")
    # downloader.download_midi(output_path)
    # assert len(os.listdir(output_path)) == 328


# cleanup
def test_cleanup():
    for Downloader in DOWNLOADERS:
        downloader = Downloader(cache_path=TEST_CACHE_PATH)
        downloader.clear_cache()
        cache_dir = os.path.join(TEST_CACHE_PATH, downloader.dataset_name)
        assert not os.path.exists(cache_dir)
    assert os.path.exists(TEST_CACHE_PATH)
    assert len(os.listdir(TEST_CACHE_PATH)) == 0
    shutil.rmtree(TEST_CACHE_PATH)


#


# def test_working_dir():
#    cwd = os.getcwd()
#    assert (os.path.basename(cwd) == 'melody_gen' and
#            os.path.exists(os.path.join(cwd, 'LICENSE'))), ("Please run tests "
#            f"from the project root directory. Currently running from {cwd}")


# def get_random_csv_filepath(seed=None, size=None, part=None):
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
# class RandomFilepath:
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


# random_path = RandomFilepath()


# def get_random_compositions(nr_comps=1, **kwargs):
#    if nr_comps == 1:
#        return Composition(csv_path=str(random_path), **kwargs)
#    return [Composition(csv_path=str(random_path), **kwargs)
#            for ii in range(nr_comps)]


# def test_data_has_been_downloaded():
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

# def test_monophonic():
#    compositions = [comp for comp in
#                    get_random_compositions(10, monophonic=True)]
#    for comp in compositions:
#        assert np.sum(comp.pianoroll[:, :, 0]) == comp.pianoroll.shape[1]


# def test_all_composition_methods_and_attributes_with_random_files():
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
