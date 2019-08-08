"""
Classes to download data from different source. Each class gets data from one
source. The base data to get is midi. To contribute a new dataset, create a
new class which extends DataDownloader, and write an accompanying test in
./tests/test_downloads.py
"""
import os
import sys
import shutil
import urllib
import warnings
import zipfile
import glob
from tqdm import tqdm



USER_HOME = os.path.expanduser('~')
DEFAULT_CACHE_PATH = os.path.join(USER_HOME, '.mdtk_cache')
DATASETS = ['PPDDSept2018Monophonic']



def download_file(source, dest, verbose=True, overwrite=None):
        """Get a file from a url and save it locally"""
        if verbose:
            print(f'Downloading {source} to {dest}')
        if os.path.exists(dest):
            if overwrite is None:
                warnings.warn(f'WARNING: {dest} already exists, not '
                              'downloading', category=UserWarning)
                return
            if not overwrite:
                raise OSError(f'{dest} already exists')
        try:
            urllib.request.urlretrieve(source, dest)
        except urllib.error.HTTPError as e:
            print(f'Url {source} does not exist', file=sys.stderr)
            raise e


def make_directory(path, overwrite=None, make_parents=True, verbose=True):
        """Convenience function to create a directory and handle cases where
        it already exists.
        
        Args
        ----
        path: str
            The path of the directory to create
        overwrite: boolean or None
            If the path already exists, if overwrite is: True - delete the 
            existing path; False - return error; None - leave the existing
            path as it is and throw a warning
        verbose: bool
            Verbosity of printing
        """
        if verbose:
            print(f'Making directory at {path}')
        
        mkdir = os.makedirs
        try:
            mkdir(path)
        except FileExistsError as e:
            if overwrite is True:
                print(f'Deleting existing directory: {path}')
                shutil.rmtree(path)
                mkdir(path)
            elif overwrite is None:
                warnings.warn(f'WARNING: {path} already exists, writing files '
                      'within here only if they do not already exist.',
                      category=UserWarning)
            elif overwrite is False:
                raise e
            else:
                raise ValueError('overwrite should be boolean or None, not '
                                 f'"{overwrite}"')


def extract_zip(zip_path, out_path, overwrite=None, verbose=True):
    """Convenience function to extract zip file to out_path.
    TODO: make work for all types of zip files."""
    if verbose:
        print(f'Extracting {zip_path} to {out_path}')
    dirname = os.path.splitext(os.path.basename(zip_path))[0]
    extracted_path = os.path.join(out_path, dirname)
    if os.path.exists(extracted_path):     
        if overwrite is True:
            warnings.warn(f'Deleting existing directory: {extracted_path}')
            shutil.rmtree(extracted_path)
        elif overwrite is None:
            warnings.warn(f'{extracted_path} already exists. Assuming this '
                          'zip has already been extracted, not extracting.',
                  category=UserWarning)
            return extracted_path
        elif overwrite is False:
            raise FileExistsError(f'{extracted_path} already exists')
    
    with zipfile.ZipFile(zip_path, 'r') as zz:
        zz.extractall(path=out_path)
    return extracted_path


def copy_file(filepath, output_path, overwrite=None):
    """Convenience function to copy a file from filepath to output_path."""
    path = os.path.join(output_path, os.path.basename(filepath))
    if os.path.exists(path):     
        if overwrite is True:
            shutil.copy(filepath, output_path)
        elif overwrite is None:
            return
        elif overwrite is False:
            raise FileExistsError(f'{path} already exists')
    else:
        shutil.copy(filepath, output_path)


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
class PPDDSept2018Monophonic(DataDownloader):
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
        base_url = ('http://tomcollinsresearch.net/research/data/mirex/'
                         'ppdd/ppdd-sep2018')
        self.download_urls = [
                os.path.join(base_url, f'PPDD-Sep2018_sym_mono_{size}.zip')
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
        
        
        

class PPDDSept2018Polyphonic(PPDDSept2018Monophonic):
    pass
    
    
    
    
    
    
    
    