"""A degrader object can be used to easily degrade data points on the fly
according to some given parameters."""
import json
import numpy as np

import mdtk.degradations as degs

class Degrader():
    """A Degrade object can be used to easily degrade musical excerpts
    on the fly."""
    
    def __init__(self, seed=None, degradations=list(degs.DEGRADATIONS.keys()),
                 degradation_dist=[1] * len(degs.DEGRADATIONS),
                 clean_prop=1 / (len(degs.DEGRADATIONS) + 1), config=None):
        """
        Create a new degrader with the given parameters.
        
        Parameters
        ----------
        seed : int
            A random seed for numpy.
            
        degradations : list(string)
            A list of the names of the degradations to use (and in what order
            to label them).
            
        degradation_dist : list(float)
            A list of the probability of each degradation given in
            degradations. This list will be normalized to sum to 1.
            
        clean_prop : float
            The proportion of degrade calls that should return clean excerpts.
            
        config : string
            The path of a json config file (created by measure_errors.py).
            If given, degradations, degradation_dist, and clean_prop will
            all be overwritten by the values in the json file.
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Load config
        if config is not None:
            with open(config, 'r') as file:
                config = json.load(file)
                
            if 'degradation_dist' in config:
                degradation_dist = np.array(config['degradation_dist'])
                degradations = degradations.DEGRADATIONS
            if 'clean_prop' in config:
                clean_prop = config['clean_prop']
                
        # Check arg validity
        assert len(degradation_dist) == len(degradations), (
            "Given degradation_dist is not the same length as degradations:"
            f"\nlen({degradation_dist}) != len({degradations})"
        )
        assert min(degradation_dist) >= 0, ("degradation_dist values must "
                                            "not be negative.")
        assert sum(degradation_dist) > 0, ("Some degradation_dist value "
                                           "must be positive.")
        assert 0 <= clean_prop <= 1, ("clean_prop must be between 0 and 1 "
                                      "(inclusive).")
        
        self.degradations = degradations
        self.degradation_dist = degradation_dist
        self.clean_prop = clean_prop
        self.failed = np.zeros(len(degradations))
        
        
    def degrade(note_df):
        """
        Degrade the given note_df.
        
        Parameters
        ----------
        note_df : pd.DataFrame
            A note_df to degrade.
            
        Returns
        -------
        degraded_df : pd.DataFrame
            A degraded version of the given note_df. If self.clean_prop > 0,
            this can be a copy of the given note_df.
            
        deg_label : int
            The label of the degradation that was performed. 0 means none,
            and larger numbers mean the degradation
            "self.degradations[deg_label+1]" was performed.
        """
        pass
    