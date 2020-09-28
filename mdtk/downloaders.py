"""
Classes to download data from different source. Each class gets data from one
source. The base data to get is midi. To contribute a new dataset, create a
new class which extends DataDownloader, and write an accompanying test in
./tests/test_downloads.py
"""
import os
import shutil
from glob import glob

from tqdm import tqdm

from mdtk.filesystem_utils import copy_file, download_file, extract_zip, make_directory

USER_HOME = os.path.expanduser("~")
DEFAULT_CACHE_PATH = os.path.join(USER_HOME, ".mdtk_cache")
DATASETS = ["PPDDSep2018Monophonic", "PPDDSep2018Polyphonic", "PianoMidi"]


# Classes =====================================================================
class DataDownloader:
    """Base class for data downloaders"""

    def __init__(self, cache_path=DEFAULT_CACHE_PATH):
        self.dataset_name = self.__class__.__name__
        self.download_urls = []
        self.midi_paths = []
        self.csv_paths = []
        #        self.downloaded = False
        #        self.extracted = False
        self.cache_path = cache_path
        self.midi_output_path = None
        self.csv_output_path = None

    def clear_cache(self):
        path = os.path.join(self.cache_path, self.dataset_name)
        if os.path.exists(path):
            shutil.rmtree(path)

    def download_midi(
        self, output_path, cache_path=None, overwrite=None, verbose=False
    ):
        """Downloads the MIDI data to output_path"""
        cache_path = self.cache_path if cache_path is None else cache_path
        raise NotImplementedError(
            "In order to download MIDI, you must implement the download_midi method."
        )

    def download_csv(self, output_path, cache_path=None, overwrite=None, verbose=False):
        """Downloads the csv data to output_path"""
        cache_path = self.cache_path if cache_path is None else cache_path
        raise NotImplementedError(
            "In order to download CSV, you must implement the download_csv method."
        )


class PPDDSep2018Monophonic(DataDownloader):
    """Patterns for Preditction Development Dataset. Monophonic data only.

    References
    ----------
    https://www.music-ir.org/mirex/wiki/2019:Patterns_for_Prediction
    """

    def __init__(
        self,
        cache_path=DEFAULT_CACHE_PATH,
        sizes=["small", "medium", "large"],
        clean=False,
    ):
        super().__init__(cache_path=cache_path)
        self.dataset_name = self.__class__.__name__
        self.base_url = (
            "http://tomcollinsresearch.net/research/data/mirex/ppdd/ppdd-sep2018"
        )
        self.download_urls = [
            f"{self.base_url}/PPDD-Sep2018_sym_mono_{size}.zip" for size in sizes
        ]
        # location of midi files relative to each unzipped directory
        #        self.midi_paths = ['prime_midi', 'cont_true_midi']
        self.midi_paths = ["prime_midi"]
        self.clean = clean

    def download_midi(
        self, output_path, cache_path=None, overwrite=None, verbose=False
    ):
        # Cleaning paths, and setting up cache dir ============================
        cache_path = self.cache_path if cache_path is None else cache_path
        base_path = os.path.join(cache_path, self.dataset_name)
        make_directory(base_path, overwrite, verbose=verbose)
        make_directory(output_path, overwrite, verbose=verbose)

        # Downloading the data ================================================
        zip_paths = []
        for url in self.download_urls:
            filename = url.split("/")[-1]
            zip_path = os.path.join(base_path, filename)
            zip_paths += [zip_path]
            download_file(url, zip_path, overwrite=overwrite, verbose=verbose)

        # Extracting data from zip files ======================================
        extracted_paths = []
        for zip_path in zip_paths:
            path = extract_zip(
                zip_path, base_path, overwrite=overwrite, verbose=verbose
            )
            extracted_paths += [path]

        # Copying midi files to output_path ===================================
        for path in extracted_paths:
            midi_paths = [
                glob(os.path.join(path, mp, "*.mid")) for mp in self.midi_paths
            ]
            midi_paths = [pp for sublist in midi_paths for pp in sublist]
            for filepath in tqdm(midi_paths, desc=f"Copying midi to {output_path}"):
                copy_file(filepath, output_path)
        self.midi_output_path = output_path

        # Delete cache ========================================================
        if self.clean:
            self.clear_cache()


class PPDDSep2018Polyphonic(PPDDSep2018Monophonic):
    """Patterns for Preditction Development Dataset. Polyphonic data only.

    References
    ----------
    https://www.music-ir.org/mirex/wiki/2019:Patterns_for_Prediction
    """

    def __init__(
        self,
        cache_path=DEFAULT_CACHE_PATH,
        sizes=["small", "medium", "large"],
        clean=False,
    ):
        super().__init__(cache_path=cache_path, sizes=sizes, clean=clean)
        self.download_urls = [
            f"{self.base_url}/PPDD-Sep2018_sym_poly_{size}.zip" for size in sizes
        ]


class PianoMidi(DataDownloader):
    """
    piano-midi dataset from http://www.piano-midi.de/
    """

    def __init__(
        self,
        cache_path=DEFAULT_CACHE_PATH,
        clean=False,
        composers=[
            "albeniz",
            "bach",
            "balakir",
            "beeth",
            "borodin",
            "brahms",
            "burgm",
            "chopin",
            "clementi",
            "debussy",
            "godowsky",
            "granados",
            "grieg",
            "haydn",
            "liszt",
            "mendelssohn",
            "moszkowski",
            "mozart",
            "muss",
            "rachmaninov",
            "ravel",
            "schubert",
            "schumann",
            "sinding",
            "tchai",
        ],
    ):
        super().__init__(cache_path=cache_path)
        self.dataset_name = self.__class__.__name__
        base_url = "http://www.piano-midi.de/zip"
        self.download_urls = []

        # Some composers don't have zip files
        non_zips = [
            "clementi",
            "godowsky",
            "moszkowski",
            "rachmaninov",
            "ravel",
            "sinding",
        ]

        for composer in composers:
            if composer in non_zips:
                self.download_urls.extend(
                    self.get_composer_urls("http://www.piano-midi.de/midis", composer)
                )
            else:
                self.download_urls.append(f"{base_url}/{composer}.zip")
        self.clean = clean

    def get_composer_urls(self, base_url, composer):
        """
        Some composers in piano-midi do not have zip files to download and
        each MIDI file must be downloaded individually. This function gets
        the urls for those MIDI files.
        """
        sub_dir = composer if composer != "rachmaninov" else "rachmaninow"

        midi_names = []
        if composer == "clementi":
            prefix = "clementi_opus36"
            for number in range(1, 7):
                for movement in range(1, 4):
                    if number == 6 and movement == 3:
                        continue
                    midi_names.append(f"{prefix}_{number}_{movement}")

        elif composer == "godowsky":
            midi_names = ["god_chpn_op10_e01", "god_alb_esp2"]

        elif composer == "moszkowski":
            midi_names = ["mos_op36_6"]

        elif composer == "rachmaninov":
            prefix = "rac_op"
            opus = [3, 23, 32, 33]
            numbers = [[2], [2, 3, 5, 7], [1, 13], [5, 6, 8]]
            for op, nos in zip(opus, numbers):
                for no in nos:
                    midi_names.append(f"{prefix}{op}_{no}")

        elif composer == "ravel":
            midi_names = [
                "rav_eau",
                "ravel_miroirs_1",
                "rav_ondi",
                "rav_gib",
                "rav_scarbo",
            ]

        elif composer == "sinding":
            midi_names = ["fruehlingsrauschen"]

        else:
            raise NotImplementedError(
                f"Downloading of composer {composer} "
                "not yet implemented in PianoMidi."
            )

        urls = [f"{base_url}/{sub_dir}/{midi}.mid" for midi in midi_names]
        return urls

    def download_midi(
        self, output_path, cache_path=None, overwrite=None, verbose=False
    ):
        # Cleaning paths, and setting up cache dir ============================
        cache_path = self.cache_path if cache_path is None else cache_path
        base_path = os.path.join(cache_path, self.dataset_name)
        midi_dl_path = os.path.join(base_path, "midis")
        make_directory(base_path, overwrite, verbose=verbose)
        make_directory(output_path, overwrite, verbose=verbose)
        make_directory(midi_dl_path, overwrite, verbose=verbose)

        # Downloading the data ================================================
        zip_paths = []
        for url in self.download_urls:
            filename = url.split("/")[-1]
            if filename.endswith(".zip"):
                path = os.path.join(base_path, filename)
                zip_paths += [path]
            else:
                path = os.path.join(midi_dl_path, filename)
            download_file(url, path, overwrite=overwrite, verbose=verbose)
        #            I removed this sleep because it makes things very slow in the case
        #            of things already having been downloaded, is it really required?
        #            time.sleep(1) # Don't want to overload the server

        # Extracting data from zip files ======================================
        extracted_paths = [midi_dl_path]
        for zip_path in zip_paths:
            zip_name = os.path.splitext(os.path.basename(zip_path))[0]
            out_path = os.path.join(base_path, zip_name)
            extract_zip(zip_path, out_path, overwrite=overwrite, verbose=verbose)
            extracted_paths += [out_path]

        # Copying midi files to output_path ===================================
        for filepath in tqdm(
            [p for path in extracted_paths for p in glob(os.path.join(path, "*.mid"))],
            desc=f"Copying midi to {output_path}: ",
        ):
            copy_file(filepath, output_path)
        self.midi_output_path = output_path

        # Delete cache ========================================================
        if self.clean:
            self.clear_cache()
