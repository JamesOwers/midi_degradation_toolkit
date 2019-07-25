"""
Classes to download data from different source. Each class gets data from one
source. The base data to get is midi. To contribute a new dataset, create a
new class which extends DataDownloader, and write an accompanying test in
./tests/test_downloads.py
"""
import os



USER_HOME = os.path.expanduser('~')
DEFAULT_CACHE_LOCATION = os.path.join(USER_HOME, '.mdtk_cache')



class DataDownloader:
    """Base class for data downloaders"""
    def __init__(cache_location=DEFAULT_CACHE_LOCATION, midi_output_loc=None,
                 csv_output_loc=None):
        self.dataset_name = ''
        self.base_url = ''
        self.is_zipped = False
        self.midi_locations = []
        self.csv_locations = []
        self.cache_location = cache_location
        self.output_loc = {'midi': midi_output_loc, 'csv': csv_output_loc}
        
    
    def get_output_and_cache_loc(self, data_type, output_loc=None,
                                 cache_loc=None):
        if output_loc is None:
            output_loc = self.output_loc[data_type]
        if cache_loc is None:
            cache_loc = self.cache_loc
        return output_loc, cache_loc
    
        
    def download_midi(self, output_loc=None, cache_loc=None):
        output_loc, cache_loc = self.get_output_and_cache_loc(
                'midi',
                output_loc,
                cache_loc
            )
        raise NotImplementedError('In order to download MIDI, you must '
                                  'implement the download_midi method.')
        
    
    def download_csv(self, output_loc=None, cache_loc=None):
        output_loc, cache_loc = self.get_output_and_cache_loc(
                'csv',
                output_loc,
                cache_loc
            )
        raise NotImplementedError('In order to download CSV, you must '
                                  'implement the download_midi method.')
        


class PPDDSept2018(DataDownloader):
    """Patterns for Preditction Development Dataset
    
    References
    ----------
    https://www.music-ir.org/mirex/wiki/2019:Patterns_for_Prediction
    """
    