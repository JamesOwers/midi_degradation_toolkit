import pandas as pd
import os
import pretty_midi
import shutil

import mdtk.midi as midi

USER_HOME = os.path.expanduser('~')
TEST_CACHE_PATH = os.path.join(USER_HOME, '.mdtk_test_cache')
MIDI_PATH = f"mdtk{os.path.sep}tests"

TEST_MID = f"{MIDI_PATH}{os.path.sep}test.mid"
ALB_MID = f"{MIDI_PATH}{os.path.sep}alb_se2.mid"


def test_midi_rounding():
    df = midi.midi_to_df(ALB_MID)


def test_df_to_csv():
    notes = []
    notes.append({'onset': 0,
                  'track': 1,
                  'pitch': 42,
                  'dur': 1})
    notes.append({'onset': 0,
                  'track': 1,
                  'pitch': 40,
                  'dur': 1})
    notes.append({'onset': 1,
                  'track': 2,
                  'pitch': 56,
                  'dur': 1})

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
            assert int(split[0]) == notes[i]['onset'], ("Onset time of "
                   f"note {i} ({notes[i]}) not equal to csv's written onset "
                   f"time of {int(split[0])}")
            assert split[1].isdigit(), f"Track {split[1]} is not an int."
            assert int(split[1]) == notes[i]['track'], ("Track of "
                   f"note {i} ({notes[i]}) not equal to csv's written track "
                   f"of {int(split[1])}")
            assert split[2].isdigit(), f"Pitch {split[2]} is not an int."
            assert int(split[2]) == notes[i]['pitch'], ("Pitch of "
                   f"note {i} ({notes[i]}) not equal to csv's written pitch "
                   f"of {int(split[2])}")
            assert int(split[3]) == notes[i]['dur'], ("Duration of "
                   f"note {i} ({notes[i]}) not equal to csv's written "
                   f"duration of {int(split[3])}")



def test_midi_to_df():
    df = midi.midi_to_df(TEST_MID)

    # This relies on pretty_midi being correct
    m = pretty_midi.PrettyMIDI(TEST_MID)
    midi_notes = []
    for i, instrument in enumerate(m.instruments):
        for note in instrument.notes:
            midi_notes.append({'onset': int(round(note.start * 1000)),
                               'track': i,
                               'pitch': note.pitch,
                               'dur': int(round(note.end * 1000) -
                                          round(note.start * 1000))})

    df_notes = df.to_dict('records')

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

    midi.midi_to_csv(TEST_MID, csv_path)

    # This relies on pretty_midi being correct
    m = pretty_midi.PrettyMIDI(TEST_MID)
    midi_notes = []
    for i, instrument in enumerate(m.instruments):
        for note in instrument.notes:
            midi_notes.append({'onset': int(round(note.start * 1000)),
                               'track': i,
                               'pitch': note.pitch,
                               'dur': int(round(note.end * 1000) -
                                          round(note.start * 1000))})

    # Check that notes were written correctly
    with open(csv_path, 'r') as file:
        for i, line in enumerate(file):
            split = line.split(',')
            note = {'onset': int(split[0]),
                    'track': int(split[1]),
                    'pitch': int(split[2]),
                    'dur': int(split[3])}
            assert note in midi_notes, (f"csv note {note} not in list " +
                                        "of MIDI notes from pretty_midi " +
                                        "(or was duplicated).")
            midi_notes.remove(note)

    # Test that all MIDI notes were in the df
    assert len(midi_notes) == 0, ("Some MIDI notes (from pretty_midi) were " +
                                  f"not found in the DataFrame: {midi_notes}")

    # Check writing without any directory
    midi.midi_to_csv(TEST_MID, 'test.csv')
    try:
        os.remove('test.csv')
    except:
        pass


def test_midi_dir_to_csv():
    midi_dir = os.path.dirname(TEST_MID)
    csv_dir = TEST_CACHE_PATH
    csv_paths = [csv_dir + os.path.sep + 'test.csv',
                 csv_dir + os.path.sep + 'test2.csv',
                 csv_dir + os.path.sep + 'alb_se2.csv']

    for csv_path in csv_paths:
        try:
            os.remove(csv_path)
        except:
            pass

    midi2_path = os.path.dirname(TEST_MID) + os.path.sep + 'test2.mid'
    shutil.copyfile(TEST_MID, midi2_path)

    midi.midi_dir_to_csv(midi_dir, csv_dir)

    os.remove(midi2_path)

    for csv_path in csv_paths:
        assert os.path.exists(csv_path), f"{csv_path} was not created."

    # This relies on pretty_midi being correct
    m = pretty_midi.PrettyMIDI(TEST_MID)
    midi_notes = []
    midi_notes2 = []
    for i, instrument in enumerate(m.instruments):
        for note in instrument.notes:
            midi_notes.append({'onset': int(round(note.start * 1000)),
                               'track': i,
                               'pitch': note.pitch,
                               'dur': int(round(note.end * 1000) -
                                          round(note.start * 1000))})
            midi_notes2.append(midi_notes[-1])

    # Check that notes were written correctly
    for csv_path, notes in zip(csv_paths, [midi_notes, midi_notes2]):
        with open(csv_path, 'r') as file:
            for i, line in enumerate(file):
                split = line.split(',')
                note = {'onset': int(split[0]),
                        'track': int(split[1]),
                        'pitch': int(split[2]),
                        'dur': int(split[3])}
                assert note in notes, (f"csv note {note} not in list " +
                                       "of MIDI notes from pretty_midi " +
                                       "(or was duplicated).")
                notes.remove(note)

    # Test that all MIDI notes were in the df
    for notes in [midi_notes, midi_notes2]:
        assert len(notes) == 0, ("Some MIDI notes (from pretty_midi) were " +
                                 f"not found in the DataFrame: {notes}")


def test_df_to_midi():
    df = pd.DataFrame({
        'onset': 0,
        'track': [0, 0, 1],
        'pitch': [10, 20, 30],
        'dur': 1000
    })

    # Test basic writing
    midi.df_to_midi(df, 'test.mid')
    assert midi.midi_to_df('test.mid').equals(df), (
        "Writing df to MIDI and reading changes df."
    )

    # Test that writing should overwrite existing notes
    df.pitch += 10
    midi.df_to_midi(df, 'test2.mid', existing_midi_path='test.mid')
    assert midi.midi_to_df('test2.mid').equals(df), (
        "Writing df to MIDI with existing MIDI does not overwrite notes."
    )

    # Test that writing skips non-overwritten notes
    midi.df_to_midi(df, 'test2.mid', existing_midi_path='test.mid',
                    excerpt_start=1000)
    expected = pd.DataFrame({
        'onset': [0, 0, 0, 1000, 1000, 1000],
        'track': [0, 0, 1, 0, 0, 1],
        'pitch': [10, 20, 30, 20, 30, 40],
        'dur': 1000
    })
    assert midi.midi_to_df('test2.mid').equals(expected), (
        "Writing to MIDI doesn't copy notes before excerpt_start"
    )

    # Test that writing skips non-overwritten notes past end
    midi.df_to_midi(df, 'test.mid', existing_midi_path='test2.mid',
                    excerpt_length=1000)
    expected = pd.DataFrame({
        'onset': [0, 0, 0, 1000, 1000, 1000],
        'track': [0, 0, 1, 0, 0, 1],
        'pitch': [20, 30, 40, 20, 30, 40],
        'dur': 1000
    })
    assert midi.midi_to_df('test.mid').equals(expected), (
        "Writing to MIDI doesn't copy notes after excerpt_length"
    )

    df.track = 2
    midi.df_to_midi(df, 'test.mid', existing_midi_path='test2.mid',
                    excerpt_length=1000)
    expected = pd.DataFrame({
        'onset': [0, 0, 0, 1000, 1000, 1000],
        'track': [2, 2, 2, 0, 0, 1],
        'pitch': [20, 30, 40, 20, 30, 40],
        'dur': 1000
    })
    assert midi.midi_to_df('test.mid').equals(expected), (
        "Writing to MIDI with extra track breaks"
    )

    # Check all non-note events
    midi_obj = pretty_midi.PrettyMIDI('test.mid')
    midi_obj.instruments[0].name = 'test'
    midi_obj.instruments[0].program = 100
    midi_obj.instruments[0].is_drum = True
    midi_obj.instruments[0].pitch_bends.append(pretty_midi.PitchBend(10, 0))
    midi_obj.instruments[0].control_changes.append(
        pretty_midi.ControlChange(10, 10, 0)
    )
    midi_obj.lyrics.append(pretty_midi.Lyric("test", 0))
    midi_obj.time_signature_changes.append(pretty_midi.TimeSignature(2, 4, 1))
    midi_obj.key_signature_changes.append(pretty_midi.KeySignature(5, 1))
    midi_obj.write('test.mid')

    midi.df_to_midi(expected, 'test2.mid', existing_midi_path='test.mid')
    assert midi.midi_to_df('test2.mid').equals(expected)

    # Check non-note events and data here
    new_midi = pretty_midi.PrettyMIDI('test2.mid')

    for instrument, new_instrument in zip(midi_obj.instruments,
                                          new_midi.instruments):
        assert instrument.name == new_instrument.name
        assert instrument.program == new_instrument.program
        assert instrument.is_drum == new_instrument.is_drum
        for pb, new_pb in zip(instrument.pitch_bends,
                              new_instrument.pitch_bends):
            assert pb.pitch == new_pb.pitch
            assert pb.time == new_pb.time
        for cc, new_cc in zip(instrument.control_changes,
                              new_instrument.control_changes):
            assert cc.number == new_cc.number
            assert cc.value == new_cc.value
            assert cc.time == new_cc.time

    for ks, new_ks in zip(midi_obj.key_signature_changes,
                          new_midi.key_signature_changes):
        assert ks.key_number == new_ks.key_number
        assert ks.time == new_ks.time

    for lyric, new_lyric in zip(midi_obj.lyrics, new_midi.lyrics):
        assert lyric.text == new_lyric.text
        assert lyric.time == new_lyric.time

    for ts, new_ts in zip(midi_obj.time_signature_changes,
                          new_midi.time_signature_changes):
        assert ts.numerator == new_ts.numerator
        assert ts.denominator == new_ts.denominator
        assert ts.time == new_ts.time

    for filename in ['test.mid', 'test2.mid']:
        try:
            os.remove(filename)
        except:
            pass


def test_csv_to_midi():
    df = pd.DataFrame({
        'onset': 0,
        'track': [0, 0, 1],
        'pitch': [10, 20, 30],
        'dur': 1000
    })
    midi.df_to_csv(df, 'test.csv')

    # Test basic writing
    midi.csv_to_midi('test.csv', 'test.mid')
    assert midi.midi_to_df('test.mid').equals(df), (
        "Writing df to MIDI and reading changes df."
    )

    # Test that writing should overwrite existing notes
    df.pitch += 10
    midi.df_to_csv(df, 'test.csv')
    midi.csv_to_midi('test.csv', 'test2.mid', existing_midi_path='test.mid')
    assert midi.midi_to_df('test2.mid').equals(df), (
        "Writing df to MIDI with existing MIDI does not overwrite notes."
    )

    # Test that writing skips non-overwritten notes
    midi.csv_to_midi('test.csv', 'test2.mid', existing_midi_path='test.mid',
                     excerpt_start=1000)
    expected = pd.DataFrame({
        'onset': [0, 0, 0, 1000, 1000, 1000],
        'track': [0, 0, 1, 0, 0, 1],
        'pitch': [10, 20, 30, 20, 30, 40],
        'dur': 1000
    })
    assert midi.midi_to_df('test2.mid').equals(expected), (
        "Writing to MIDI doesn't copy notes before excerpt_start"
    )

    # Test that writing skips non-overwritten notes past end
    midi.csv_to_midi('test.csv', 'test.mid', existing_midi_path='test2.mid',
                    excerpt_length=1000)
    expected = pd.DataFrame({
        'onset': [0, 0, 0, 1000, 1000, 1000],
        'track': [0, 0, 1, 0, 0, 1],
        'pitch': [20, 30, 40, 20, 30, 40],
        'dur': 1000
    })
    assert midi.midi_to_df('test.mid').equals(expected), (
        "Writing to MIDI doesn't copy notes after excerpt_length"
    )

    df.track = 2
    midi.df_to_csv(df, 'test.csv')
    midi.csv_to_midi('test.csv', 'test.mid', existing_midi_path='test2.mid',
                     excerpt_length=1000)
    expected = pd.DataFrame({
        'onset': [0, 0, 0, 1000, 1000, 1000],
        'track': [2, 2, 2, 0, 0, 1],
        'pitch': [20, 30, 40, 20, 30, 40],
        'dur': 1000
    })
    assert midi.midi_to_df('test.mid').equals(expected), (
        "Writing to MIDI with extra track breaks"
    )

    for filename in ['test.mid', 'test2.mid', 'test.csv']:
        try:
            os.remove(filename)
        except:
            pass
