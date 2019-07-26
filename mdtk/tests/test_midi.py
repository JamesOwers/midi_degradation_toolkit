import pandas as pd
import os
import pretty_midi

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
    
    # Check that notes were written correctly
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
            
            
            
def test_midi_to_df():
    midi_path = f"mdtk{os.path.sep}tests{os.path.sep}test.mid"
    df = midi.midi_to_df(midi_path)
    
    # This relies on pretty_midi being correct
    m = pretty_midi.PrettyMIDI(midi_path)
    midi_notes = []
    for i, instrument in enumerate(m.instruments):
        for note in instrument.notes:
            midi_notes.append({'onset': note.start * 1000,
                               'track': i,
                               'pitch': note.pitch,
                               'dur': (note.end - note.start) * 1000})
    
    df_notes = []
    for index, df_note in df.iterrows():
        # There must be a better way to do this
        df_notes.append({'onset': df_note['onset'],
                         'track': df_note['track'],
                         'pitch': df_note['pitch'],
                         'dur': df_note['dur']})
    
    # Test sorting
    for prev_note, next_note in zip(df_notes[:-1], df_notes[1:]):
        assert (prev_note['onset'] < next_note['onset'] or
                (prev_note['onset'] == next_note['onset'] and
                 prev_note['track'] < next_note['track']) or
                (prev_note['onset'] == next_note['onset'] and
                 prev_note['track'] == next_note['track'] and
                 prev_note['pitch'] < next_note['pitch']) or
                (prev_note['onset'] == next_note['onset'] and
                 prev_note['track'] == next_note['track'] and
                 prev_note['pitch'] == next_note['pitch'] and
                 prev_note['dur'] < next_note['dur'])), ("DataFrame sorting " +
                f"incorrect. {prev_note} is before {next_note}")
                
    # Test that all notes df notes are in the MIDI
    for df_note in df_notes:
        assert df_note in midi_notes, (f"DataFrame note {df_note} not in " +
                                       "list of MIDI notes from pretty_midi " +
                                       "(or was duplicated).")
        midi_notes.remove(df_note)
        
    # Test that all MIDI notes were in the df
    assert len(midi_notes) == 0, ("Some MIDI notes (from pretty_midi) were " +
                                  f"not found in the DataFrame: {midi_notes}")
    