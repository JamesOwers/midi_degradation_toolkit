import pandas as pd
import os
import pretty_midi
import shutil

import mdtk.midi as midi

USER_HOME = os.path.expanduser('~')
TEST_CACHE_PATH = os.path.join(USER_HOME, '.mdtk_test_cache')
MIDI_PATH = f"mdtk{os.path.sep}tests{os.path.sep}test.mid"

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
            assert split[1].isdigit(), f"Track {split[1]} is not an int."
            assert int(split[1]) == notes[i]['track'], ("Track of " +
                   f"note {i} ({notes[i]}) not equal to csv's written track " +
                   f"of {int(split[1])}")
            assert split[2].isdigit(), f"Pitch {split[2]} is not an int."
            assert int(split[2]) == notes[i]['pitch'], ("Pitch of " +
                   f"note {i} ({notes[i]}) not equal to csv's written pitch " +
                   f"of {int(split[2])}")
            assert float(split[3]) == notes[i]['dur'], ("Duration of " +
                   f"note {i} ({notes[i]}) not equal to csv's written " +
                   f"duration of {float(split[3])}")
            
            
            
def test_midi_to_df():
    df = midi.midi_to_df(MIDI_PATH)
    
    # This relies on pretty_midi being correct
    m = pretty_midi.PrettyMIDI(MIDI_PATH)
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
    
    
    
def test_midi_to_csv():
    # This method is just calls to midi_to_df and df_to_csv
    csv_path = TEST_CACHE_PATH + os.path.sep + 'test.csv'
    
    try:
        os.remove(csv_path)
    except:
        pass
    
    midi.midi_to_csv(MIDI_PATH, csv_path)
    
    # This relies on pretty_midi being correct
    m = pretty_midi.PrettyMIDI(MIDI_PATH)
    midi_notes = []
    for i, instrument in enumerate(m.instruments):
        for note in instrument.notes:
            midi_notes.append({'onset': note.start * 1000,
                               'track': i,
                               'pitch': note.pitch,
                               'dur': (note.end - note.start) * 1000})
            
    # Check that notes were written correctly
    with open(csv_path, 'r') as file:
        for i, line in enumerate(file):
            split = line.split(',')
            note = {'onset': float(split[0]),
                    'track': int(split[1]),
                    'pitch': int(split[2]),
                    'dur': float(split[3])}
            assert note in midi_notes, (f"csv note {note} not in list " +
                                        "of MIDI notes from pretty_midi " +
                                        "(or was duplicated).")
            midi_notes.remove(note)
            
    # Test that all MIDI notes were in the df
    assert len(midi_notes) == 0, ("Some MIDI notes (from pretty_midi) were " +
                                  f"not found in the DataFrame: {midi_notes}")
    
    
def test_midi_dir_to_csv():
    midi_dir = os.path.dirname(MIDI_PATH)
    csv_dir = TEST_CACHE_PATH
    csv_paths = [csv_dir + os.path.sep + 'test.csv',
                 csv_dir + os.path.sep + 'test2.csv']
    
    for csv_path in csv_paths:
        try:
            os.remove(csv_path)
        except:
            pass
    
    midi2_path = os.path.dirname(MIDI_PATH) + os.path.sep + 'test2.mid'
    shutil.copyfile(MIDI_PATH, midi2_path)
    
    midi.midi_dir_to_csv(midi_dir, csv_dir)
    
    os.remove(midi2_path)
    
    for csv_path in csv_paths:
        assert os.path.exists(csv_path), f"{csv_path} was not created."
    
    # This relies on pretty_midi being correct
    m = pretty_midi.PrettyMIDI(MIDI_PATH)
    midi_notes = []
    midi_notes2 = []
    for i, instrument in enumerate(m.instruments):
        for note in instrument.notes:
            midi_notes.append({'onset': note.start * 1000,
                               'track': i,
                               'pitch': note.pitch,
                               'dur': (note.end - note.start) * 1000})
            midi_notes2.append(midi_notes[-1])
            
    # Check that notes were written correctly
    for csv_path, notes in zip(csv_paths, [midi_notes, midi_notes2]):
        with open(csv_path, 'r') as file:
            for i, line in enumerate(file):
                split = line.split(',')
                note = {'onset': float(split[0]),
                        'track': int(split[1]),
                        'pitch': int(split[2]),
                        'dur': float(split[3])}
                assert note in notes, (f"csv note {note} not in list " +
                                       "of MIDI notes from pretty_midi " +
                                       "(or was duplicated).")
                notes.remove(note)
            
    # Test that all MIDI notes were in the df
    for notes in [midi_notes, midi_notes2]:
        assert len(notes) == 0, ("Some MIDI notes (from pretty_midi) were " +
                                 f"not found in the DataFrame: {notes}")
    