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
        
        comp2 = deg.pitch_shift(comp, min_pitch=100 * i, max_pitch=100 * (i + 1))
        
        equal = (comp2.note_df == BASIC_DF)
        
        # Check that only things that should have changed have changed
        assert equal['onset'].all(), "Pitch shift changed some onset time."
        assert equal['track'].all(), "Pitch shift changed some track."
        assert equal['dur'].all(), "Pitch shift changed some duration."
        assert (1 - equal['pitch']).sum() == 1, ("Pitch shift did not change "
                                                 "exactly one pitch.")
        
        # Check that changed pitch is within given range
        changed_pitch = comp2.note_df[(comp2.note_df['pitch'] !=
                                       BASIC_DF['pitch'])]['pitch'].iloc[0]
        assert 100 * i <= changed_pitch <= 100 * (i + 1), (f"Pitch {changed_pitch} "
                                                           f"outside of range [{100 * i}"
                                                           f", {100 * (i + 1)}]")
        
        # Check a simple setting of the distribution parameter
        distribution = np.zeros(3)
        sample = np.random.randint(3)
        if sample == 1:
            sample = 2
        distribution[sample] = 1
        correct_diff = 1 - sample
        
        comp2 = deg.pitch_shift(comp, distribution=distribution)
        
        not_equal = (comp2.note_df['pitch'] != BASIC_DF['pitch'])
        changed_pitch = comp2.note_df[not_equal]['pitch'].iloc[0]
        original_pitch = BASIC_DF[not_equal]['pitch'].iloc[0]
        diff = original_pitch - changed_pitch
        
        assert diff == correct_diff, (f"Pitch difference {diff} is not equal to correct "
                                      f" difference {correct_diff} with distribution = "
                                      f"length 50 list of 0's with 1 at index {sample}.")
        
    # Check for distribution warnings
    with pytest.warns(UserWarning, match=re.escape('WARNING: distribution contains only '
                                                   '0s after setting middle value to 0. '
                                                   'Returning None.')):
        comp2 = deg.pitch_shift(comp, distribution=[0, 1, 0])
        assert comp2 == None, "Pitch shifting with distribution of 0s returned something."
        
    with pytest.warns(UserWarning, match=re.escape('WARNING: No valid pitches to shift '
                                                   'given min_pitch')):
        comp2 = deg.pitch_shift(comp, min_pitch=-50, max_pitch=-20, distribution=[1, 0, 1])
        assert comp2 == None, "Pitch shifting with invalid distribution returned something."