"""
Classes to download data from different source. Each class gets data from one
source. The base data to get is midi. To contribute a new dataset, create a
new class which extends DataDownloader, and write an accompanying test in
./tests/test_downloads.py
"""
import os
import shutil
import glob
from tqdm import tqdm
from mdtk.filesystem_utils import (download_file, make_directory, extract_zip,
                                   copy_file)


USER_HOME = os.path.expanduser('~')
DEFAULT_CACHE_PATH = os.path.join(USER_HOME, '.mdtk_cache')
DATASETS = ['PPDDSep2018Monophonic', 'PPDDSep2018Polyphonic']



# Classes =====================================================================
# TODO: make attributes useful to users standard e.g. beat-aligned=True/False
# TODO: some things are likely to be important re preprocessing e.g. the unit
#       for the onset and duration of notes. Add these as attributes too. 
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
#        warnings.warn(f'Deleting existing directory: {path}')
        shutil.rmtree(path)
#        self.downloaded = False
#        self.extracted = False
    
    
    def download_midi(self, output_path, cache_path=None):
        """Downloads the MIDI data to output_path"""
        cache_path = self.cache_path if cache_path is None else cache_path
        raise NotImplementedError('In order to download MIDI, you must '
                                  'implement the download_midi method.')
        
    
    def download_csv(self, output_path, cache_path=None):
        """Downloads the csv data to output_path"""
        cache_path = self.cache_path if cache_path is None else cache_path
        raise NotImplementedError('In order to download CSV, you must '
                                  'implement the download_csv method.')
        
        

# TODO: maybe make a base PPDD class and extend this for various options
class PPDDSep2018Monophonic(DataDownloader):
    """Patterns for Preditction Development Dataset. Monophonic data only.
    
    References
    ----------
    https://www.music-ir.org/mirex/wiki/2019:Patterns_for_Prediction
    """
    # TODO: add 'sample_size', to allow only a small random sample of the
    #       total midi files to be copied to the output
    def __init__(self, cache_path=DEFAULT_CACHE_PATH,
                 sizes=['small', 'medium', 'large'], clean=False):
        super().__init__(cache_path = cache_path)
        self.dataset_name = self.__class__.__name__
        self.base_url = ('http://tomcollinsresearch.net/research/data/mirex/'
                         'ppdd/ppdd-sep2018')
        self.download_urls = [
                os.path.join(self.base_url,
                             f'PPDD-Sep2018_sym_mono_{size}.zip')
                for size in sizes
            ]
        # location of midi files relative to each unzipped directory
#        self.midi_paths = ['prime_midi', 'cont_true_midi']
        self.midi_paths = ['prime_midi']
        self.clean = clean
        
    
    def download_midi(self, output_path, cache_path=None, overwrite=None):
        # Cleaning paths, and setting up cache dir ============================
        cache_path = self.cache_path if cache_path is None else cache_path
        base_path = os.path.join(cache_path, self.dataset_name)
        make_directory(base_path, overwrite)
        make_directory(output_path, overwrite)
        
        # Downloading the data ================================================
        zip_paths = []
        for url in self.download_urls:
            filename = os.path.basename(url)
            zip_path = os.path.join(base_path, filename)
            zip_paths += [zip_path]
            download_file(url, zip_path, overwrite=overwrite)
        
        # Extracting data from zip files ======================================
        extracted_paths = []
        for zip_path in zip_paths:
            path = extract_zip(zip_path, base_path, overwrite=overwrite)
            extracted_paths += [path]
        
        # Copying midi files to output_path ===================================
        for path in extracted_paths:
            midi_paths = [glob.glob(os.path.join(path, mp, '*.mid')) for mp
                          in self.midi_paths]
            midi_paths = [pp for sublist in midi_paths for pp in sublist]
            for filepath in tqdm(midi_paths, 
                                 desc=f"Copying midi from {path}: "):
                copy_file(filepath, output_path)
        self.midi_output_path = output_path
        
        # Delete cache ========================================================
        if self.clean:
            self.clear_cache()
        


class PPDDSep2018Polyphonic(PPDDSep2018Monophonic):
    """Patterns for Preditction Development Dataset. Monophonic data only.
    
    References
    ----------
    https://www.music-ir.org/mirex/wiki/2019:Patterns_for_Prediction
    """
    # TODO: add 'sample_size', to allow only a small random sample of the
    #       total midi files to be copied to the output
    def __init__(self, cache_path=DEFAULT_CACHE_PATH,
                 sizes=['small', 'medium', 'large'], clean=False):
        super().__init__(cache_path=cache_path, sizes=sizes, clean=clean)
        self.download_urls = [
                os.path.join(self.base_url,
                             f'PPDD-Sep2018_sym_poly_{size}.zip')
                for size in sizes
            ]
    
    
    
    
    
    