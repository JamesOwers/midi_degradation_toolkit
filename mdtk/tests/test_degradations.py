import pandas as pd
import numpy as np
import re
import pytest

import mdtk.data_structures as ds
import mdtk.degradations as deg

EMPTY_DF = pd.DataFrame({
    'onset': [],
    'track' : [],
    'pitch': [],
    'dur': []
})

BASIC_DF = pd.DataFrame({
    'onset': [0, 100, 200, 200],
    'track': [0, 1, 0, 1],
    'pitch': [10, 20, 30, 40],
    'dur': [100, 100, 100, 100]
})

def test_pitch_shift():
    comp = ds.Composition(EMPTY_DF)
    
    with pytest.warns(UserWarning, match=re.escape("WARNING: No notes to pitch "
                                                   "shift. Returning None.")):
        assert deg.pitch_shift(comp) == None, ("Pitch shifting with empty data "
                                               "frame did not return None.")
    
    comp = ds.Composition(BASIC_DF)
    
    # Deterministic testing
    for i in range(2):
        comp2 = deg.pitch_shift(comp, seed=1)
    
        basic_res = pd.DataFrame({'onset': [0, 100, 200, 200],
                                  'track': [0, 1, 0, 1],
                                  'pitch': [10, 107, 30, 40],
                                  'dur': [100, 100, 100, 100]})
        
        assert (comp2.note_df == basic_res).all().all(), (f"Pitch shifting \n{BASIC_DF}\n"
                                                          f"resulted in \n{comp2.note_df}\n"
                                                          f"instead of \n{basic_res}")
        
        changed = comp != comp2 and (BASIC_DF != comp2.note_df).any().any()
        assert changed, "Composition or note_df was not cloned."
        
    
    # Truly random testing
    for i in range(10):
        np.random.seed()
        
        comp2 = deg.pitch_shift(comp)
        
        equal = (comp2.note_df == BASIC_DF)
        
        assert equal['onset'].all(), "Pitch shift changed some onset time."
        assert equal['track'].all(), "Pitch shift changed some track."
        assert equal['dur'].all(), "Pitch shift changed some duration."
        assert (1 - equal['pitch']).sum() == 1, ("Pitch shift did not change "
                                                 "exactly one pitch.")