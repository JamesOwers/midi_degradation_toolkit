import pandas as pd
import os

import mdtk.midi as midi

USER_HOME = os.path.expanduser('~')
TEST_CACHE_PATH = os.path.join(USER_HOME, '.mdtk_test_cache')

def test_df_to_csv():
    notes = []
    notes.append({'onset': 0.0,
                  'track': 1,
                  'pitch': 42,
                  'dur': 1.5})
    notes.append({'onset': 0.3,
                  'track': 1,
                  'pitch': 40,
                  'dur': 1.07})
    notes.append({'onset': 0.3,
                  'track': 2,
                  'pitch': 56,
                  'dur': 1.11})
    
    df = pd.DataFrame(notes)
    
    csv_name = TEST_CACHE_PATH + os.path.sep + 'test.csv'
    try:
        os.remove(csv_name)
    except:
        pass
    
    midi.df_to_csv(df, csv_name)
    
    assert os.path.exists(csv_name), ('No csv created.')
    
    with open(csv_name, 'r') as file:
        for i, line in enumerate(file):
            split = line.split(',')
            assert float(split[0]) == notes[i]['onset'], ("Onset time of " +
                   f"note {i} ({notes[i]}) not equal to csv's written onset " +
                   f"time of {float(split[0])}")
            assert int(split[1]) == notes[i]['track'], ("Track of " +
                   f"note {i} ({notes[i]}) not equal to csv's written track " +
                   f"of {int(split[1])}")
            assert int(split[2]) == notes[i]['pitch'], ("Pitch of " +
                   f"note {i} ({notes[i]}) not equal to csv's written pitch " +
                   f"of {int(split[2])}")
            assert float(split[3]) == notes[i]['dur'], ("Duration of " +
                   f"note {i} ({notes[i]}) not equal to csv's written " +
                   f"duration of {float(split[3])}")