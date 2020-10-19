import itertools
import os
import shutil

import pandas as pd
import pretty_midi

import mdtk.fileio as fileio
from mdtk.df_utils import NOTE_DF_SORT_ORDER, clean_df
from mdtk.tests.test_df_utils import CLEAN_INPUT_DF, CLEAN_RES_DFS

USER_HOME = os.path.expanduser("~")
TEST_CACHE_PATH = os.path.join(USER_HOME, ".mdtk_test_cache")
MIDI_PATH = f"mdtk{os.path.sep}tests"

TEST_MID = f"{MIDI_PATH}{os.path.sep}test.mid"
ALB_MID = f"{MIDI_PATH}{os.path.sep}alb_se2.mid"


def test_midi_rounding():
    _ = fileio.midi_to_df(ALB_MID)


def test_csv_to_df():
    csv_path = os.path.join(TEST_CACHE_PATH, "test.csv")
    fileio.df_to_csv(CLEAN_INPUT_DF, csv_path)

    # Check clean_df args
    for (track, overlap) in itertools.product([False, True], repeat=2):
        kwargs = {"single_track": track, "non_overlapping": overlap}

        correct = CLEAN_RES_DFS[track][overlap]
        res = fileio.csv_to_df(csv_path, **kwargs)

        assert res.equals(correct), f"csv_to_df result incorrect with args={kwargs}"

    # Check inducing velocity = 100 if not present in csv
    CLEAN_INPUT_DF[NOTE_DF_SORT_ORDER[:-1]].to_csv(csv_path, index=None, header=False)

    res = fileio.csv_to_df(csv_path)
    correct = CLEAN_RES_DFS[False][False]
    assert res[NOTE_DF_SORT_ORDER[:-1]].equals(correct[NOTE_DF_SORT_ORDER[:-1]])
    assert all(res["velocity"] == 100)
    assert res["velocity"].dtype == "int64"


def test_df_to_csv():
    notes = []
    notes.append({"onset": 0, "track": 1, "pitch": 42, "dur": 1, "velocity": 10})
    notes.append({"onset": 0, "track": 1, "pitch": 40, "dur": 1, "velocity": 20})
    notes.append({"onset": 1, "track": 2, "pitch": 56, "dur": 5, "velocity": 30})

    df = pd.DataFrame(notes)

    csv_name = os.path.join(TEST_CACHE_PATH, "tmp_dir", "test.csv")
    try:
        os.removedirs(csv_name)
    except Exception:
        pass

    fileio.df_to_csv(df, csv_name)

    assert os.path.exists(csv_name), "No csv created."

    # Check that notes were written correctly
    with open(csv_name, "r") as csv_file:
        for i, line in enumerate(csv_file):
            split = line.split(",")
            assert int(split[0]) == notes[i]["onset"], (
                "Onset time of "
                f"note {i} ({notes[i]}) not equal to csv's written onset "
                f"time of {int(split[0])}"
            )
            assert split[1].isdigit(), f"Track {split[1]} is not an int."
            assert int(split[1]) == notes[i]["track"], (
                "Track of "
                f"note {i} ({notes[i]}) not equal to csv's written track "
                f"of {int(split[1])}"
            )
            assert split[2].isdigit(), f"Pitch {split[2]} is not an int."
            assert int(split[2]) == notes[i]["pitch"], (
                "Pitch of "
                f"note {i} ({notes[i]}) not equal to csv's written pitch "
                f"of {int(split[2])}"
            )
            assert int(split[3]) == notes[i]["dur"], (
                "Duration of "
                f"note {i} ({notes[i]}) not equal to csv's written "
                f"duration of {int(split[3])}"
            )
            assert int(split[4]) == notes[i]["velocity"], (
                "Velocity of "
                f"note {i} ({notes[i]}) not equal to csv's written "
                f"velocity of {int(split[4])}"
            )

    # Check writing without any directory
    fileio.df_to_csv(df, "test.csv")
    try:
        os.remove("test.csv")
    except Exception:
        pass

    try:
        os.removedirs(csv_name)
    except Exception:
        pass


def test_midi_to_df():
    df = fileio.midi_to_df(TEST_MID)

    # This relies on pretty_midi being correct
    m = pretty_midi.PrettyMIDI(TEST_MID)
    midi_notes = []
    for i, instrument in enumerate(m.instruments):
        for note in instrument.notes:
            midi_notes.append(
                {
                    "onset": int(round(note.start * 1000)),
                    "track": i,
                    "pitch": note.pitch,
                    "dur": int(round(note.end * 1000) - round(note.start * 1000)),
                    "velocity": note.velocity,
                }
            )
    midi_df = pd.DataFrame(midi_notes)

    df_notes = df.to_dict("records")

    # Test that all notes df notes are in the MIDI
    for df_note in df_notes:
        assert df_note in midi_notes, (
            f"DataFrame note {df_note} not in "
            + "list of MIDI notes from pretty_midi "
            + "(or was duplicated)."
        )
        midi_notes.remove(df_note)

    # Test that all MIDI notes were in the df
    assert len(midi_notes) == 0, (
        "Some MIDI notes (from pretty_midi) were "
        + f"not found in the DataFrame: {midi_notes}"
    )

    # Check clean_df (args, sorting)
    for (track, overlap) in itertools.product([False, True], repeat=2):
        kwargs = {"single_track": track, "non_overlapping": overlap}

        correct = clean_df(midi_df, **kwargs)
        res = fileio.midi_to_df(TEST_MID, **kwargs)

        assert res.equals(
            correct
        ), f"csv_to_midi not using args correctly with args={kwargs}"


def test_midi_to_csv():
    # This method is just calls to midi_to_df and df_to_csv
    csv_path = TEST_CACHE_PATH + os.path.sep + "test.csv"

    try:
        os.remove(csv_path)
    except Exception:
        pass

    fileio.midi_to_csv(TEST_MID, csv_path)

    # This relies on pretty_midi being correct
    m = pretty_midi.PrettyMIDI(TEST_MID)
    midi_notes = []
    for i, instrument in enumerate(m.instruments):
        for note in instrument.notes:
            midi_notes.append(
                {
                    "onset": int(round(note.start * 1000)),
                    "track": i,
                    "pitch": note.pitch,
                    "dur": int(round(note.end * 1000) - round(note.start * 1000)),
                    "velocity": note.velocity,
                }
            )

    # Check that notes were written correctly
    with open(csv_path, "r") as file:
        for i, line in enumerate(file):
            split = line.split(",")
            note = {
                "onset": int(split[0]),
                "track": int(split[1]),
                "pitch": int(split[2]),
                "dur": int(split[3]),
                "velocity": int(split[4]),
            }
            assert note in midi_notes, (
                f"csv note {note} not in list "
                + "of MIDI notes from pretty_midi "
                + "(or was duplicated)."
            )
            midi_notes.remove(note)

    # Test that all MIDI notes were in the df
    assert len(midi_notes) == 0, (
        "Some MIDI notes (from pretty_midi) were "
        f"not found in the DataFrame: {midi_notes}"
    )

    # Some less robust tests regarding single_track and non_overlapping
    # (Robust versions will be in the *_to_df and df_to_* functions)
    midi_path = TEST_MID
    csv_path = "test.csv"
    for (track, overlap) in itertools.product([True, False], repeat=2):
        kwargs = {"single_track": track, "non_overlapping": overlap}
        fileio.midi_to_csv(midi_path, csv_path, **kwargs)
        df = fileio.csv_to_df(csv_path, **kwargs)
        assert df.equals(
            fileio.midi_to_df(midi_path, **kwargs)
        ), "midi_to_csv not using single_track and non_overlapping correctly."

    # Check writing without any directory
    fileio.midi_to_csv(TEST_MID, "test.csv")
    try:
        os.remove("test.csv")
    except Exception:
        pass


def test_midi_dir_to_csv():
    basenames = ["test", "test2", "alb_se2"]
    midi_dir = os.path.dirname(TEST_MID)
    midi_paths = [os.path.join(midi_dir, f"{name}.mid") for name in basenames]
    csv_dir = TEST_CACHE_PATH
    csv_paths = [os.path.join(csv_dir, f"{name}.csv") for name in basenames]

    for csv_path in csv_paths:
        try:
            os.remove(csv_path)
        except Exception:
            pass

    midi2_path = os.path.dirname(TEST_MID) + os.path.sep + "test2.mid"
    shutil.copyfile(TEST_MID, midi2_path)

    fileio.midi_dir_to_csv(midi_dir, csv_dir)

    for csv_path in csv_paths:
        assert os.path.exists(csv_path), f"{csv_path} was not created."

    # This relies on pretty_midi being correct
    m = pretty_midi.PrettyMIDI(TEST_MID)
    midi_notes = []
    midi_notes2 = []
    for i, instrument in enumerate(m.instruments):
        for note in instrument.notes:
            midi_notes.append(
                {
                    "onset": int(round(note.start * 1000)),
                    "track": i,
                    "pitch": note.pitch,
                    "dur": int(round(note.end * 1000) - round(note.start * 1000)),
                    "velocity": note.velocity,
                }
            )
            midi_notes2.append(midi_notes[-1])

    # Check that notes were written correctly
    for csv_path, notes in zip(csv_paths, [midi_notes, midi_notes2]):
        with open(csv_path, "r") as file:
            for i, line in enumerate(file):
                split = line.split(",")
                note = {
                    "onset": int(split[0]),
                    "track": int(split[1]),
                    "pitch": int(split[2]),
                    "dur": int(split[3]),
                    "velocity": int(split[4]),
                }
                assert note in notes, (
                    f"csv note {note} not in list "
                    + "of MIDI notes from pretty_midi "
                    + "(or was duplicated)."
                )
                notes.remove(note)

    # Test that all MIDI notes were in the df
    for notes in [midi_notes, midi_notes2]:
        assert len(notes) == 0, (
            "Some MIDI notes (from pretty_midi) were "
            + f"not found in the DataFrame: {notes}"
        )

    # Some less robust tests regarding single_track and non_overlapping
    # (Robust versions will be in the *_to_df and df_to_* functions)
    for (track, overlap) in itertools.product([True, False], repeat=2):
        kwargs = {"single_track": track, "non_overlapping": overlap}
        fileio.midi_dir_to_csv(midi_dir, csv_dir, **kwargs)
        for midi_path, csv_path in zip(midi_paths, csv_paths):
            df = fileio.csv_to_df(csv_path, **kwargs)
            assert df.equals(fileio.midi_to_df(midi_path, **kwargs)), (
                "midi_dir_to_csv not using single_track and non_overlapping "
                "correctly."
            )

    os.remove(midi2_path)


def test_df_to_midi():
    df = pd.DataFrame(
        {
            "onset": 0,
            "track": [0, 0, 1],
            "pitch": [10, 20, 30],
            "dur": 1000,
            "velocity": 50,
        }
    )

    # Test basic writing
    fileio.df_to_midi(df, "test.mid")
    assert fileio.midi_to_df("test.mid").equals(
        df
    ), "Writing df to MIDI and reading changes df."

    # Test that writing should overwrite existing notes
    df.pitch += 10
    fileio.df_to_midi(df, "test2.mid", existing_midi_path="test.mid")
    assert fileio.midi_to_df("test2.mid").equals(
        df
    ), "Writing df to MIDI with existing MIDI does not overwrite notes."

    # Test that writing skips non-overwritten notes
    fileio.df_to_midi(
        df, "test2.mid", existing_midi_path="test.mid", excerpt_start=1000
    )
    expected = pd.DataFrame(
        {
            "onset": [0, 0, 0, 1000, 1000, 1000],
            "track": [0, 0, 1, 0, 0, 1],
            "pitch": [10, 20, 30, 20, 30, 40],
            "dur": 1000,
            "velocity": 50,
        }
    )
    assert fileio.midi_to_df("test2.mid").equals(
        expected
    ), "Writing to MIDI doesn't copy notes before excerpt_start"

    # Test that writing skips non-overwritten notes past end
    fileio.df_to_midi(
        df, "test.mid", existing_midi_path="test2.mid", excerpt_length=1000
    )
    expected = pd.DataFrame(
        {
            "onset": [0, 0, 0, 1000, 1000, 1000],
            "track": [0, 0, 1, 0, 0, 1],
            "pitch": [20, 30, 40, 20, 30, 40],
            "dur": 1000,
            "velocity": 50,
        }
    )
    assert fileio.midi_to_df("test.mid").equals(
        expected
    ), "Writing to MIDI doesn't copy notes after excerpt_length"

    df.track = 2
    fileio.df_to_midi(
        df, "test.mid", existing_midi_path="test2.mid", excerpt_length=1000
    )
    expected = pd.DataFrame(
        {
            "onset": [0, 0, 0, 1000, 1000, 1000],
            "track": [2, 2, 2, 0, 0, 1],
            "pitch": [20, 30, 40, 20, 30, 40],
            "dur": 1000,
            "velocity": 50,
        }
    )
    assert fileio.midi_to_df("test.mid").equals(
        expected
    ), "Writing to MIDI with extra track breaks"

    # Check all non-note events
    midi_obj = pretty_midi.PrettyMIDI("test.mid")
    midi_obj.instruments[0].name = "test"
    midi_obj.instruments[0].program = 100
    midi_obj.instruments[0].is_drum = True
    midi_obj.instruments[0].pitch_bends.append(pretty_midi.PitchBend(10, 0))
    midi_obj.instruments[0].control_changes.append(pretty_midi.ControlChange(10, 10, 0))
    midi_obj.lyrics.append(pretty_midi.Lyric("test", 0))
    midi_obj.time_signature_changes.append(pretty_midi.TimeSignature(2, 4, 1))
    midi_obj.key_signature_changes.append(pretty_midi.KeySignature(5, 1))
    midi_obj.write("test.mid")

    fileio.df_to_midi(expected, "test2.mid", existing_midi_path="test.mid")
    assert fileio.midi_to_df("test2.mid").equals(expected)

    # Check non-note events and data here
    new_midi = pretty_midi.PrettyMIDI("test2.mid")

    for instrument, new_instrument in zip(midi_obj.instruments, new_midi.instruments):
        assert instrument.name == new_instrument.name
        assert instrument.program == new_instrument.program
        assert instrument.is_drum == new_instrument.is_drum
        for pb, new_pb in zip(instrument.pitch_bends, new_instrument.pitch_bends):
            assert pb.pitch == new_pb.pitch
            assert pb.time == new_pb.time
        for cc, new_cc in zip(
            instrument.control_changes, new_instrument.control_changes
        ):
            assert cc.number == new_cc.number
            assert cc.value == new_cc.value
            assert cc.time == new_cc.time

    for ks, new_ks in zip(
        midi_obj.key_signature_changes, new_midi.key_signature_changes
    ):
        assert ks.key_number == new_ks.key_number
        assert ks.time == new_ks.time

    for lyric, new_lyric in zip(midi_obj.lyrics, new_midi.lyrics):
        assert lyric.text == new_lyric.text
        assert lyric.time == new_lyric.time

    for ts, new_ts in zip(
        midi_obj.time_signature_changes, new_midi.time_signature_changes
    ):
        assert ts.numerator == new_ts.numerator
        assert ts.denominator == new_ts.denominator
        assert ts.time == new_ts.time

    for filename in ["test.mid", "test2.mid"]:
        try:
            os.remove(filename)
        except Exception:
            pass


def test_csv_to_midi():
    df = pd.DataFrame(
        {
            "onset": 0,
            "track": [0, 0, 1],
            "pitch": [10, 20, 30],
            "dur": 1000,
            "velocity": 50,
        }
    )
    fileio.df_to_csv(df, "test.csv")

    # Test basic writing
    fileio.csv_to_midi("test.csv", "test.mid")
    assert fileio.midi_to_df("test.mid").equals(
        df
    ), "Writing df to MIDI and reading changes df."

    # Test that writing should overwrite existing notes
    df.pitch += 10
    fileio.df_to_csv(df, "test.csv")
    fileio.csv_to_midi("test.csv", "test2.mid", existing_midi_path="test.mid")
    assert fileio.midi_to_df("test2.mid").equals(
        df
    ), "Writing df to MIDI with existing MIDI does not overwrite notes."

    # Test that writing skips non-overwritten notes
    fileio.csv_to_midi(
        "test.csv", "test2.mid", existing_midi_path="test.mid", excerpt_start=1000
    )
    expected = pd.DataFrame(
        {
            "onset": [0, 0, 0, 1000, 1000, 1000],
            "track": [0, 0, 1, 0, 0, 1],
            "pitch": [10, 20, 30, 20, 30, 40],
            "dur": 1000,
            "velocity": 50,
        }
    )
    assert fileio.midi_to_df("test2.mid").equals(
        expected
    ), "Writing to MIDI doesn't copy notes before excerpt_start"

    # Test that writing skips non-overwritten notes past end
    fileio.csv_to_midi(
        "test.csv", "test.mid", existing_midi_path="test2.mid", excerpt_length=1000
    )
    expected = pd.DataFrame(
        {
            "onset": [0, 0, 0, 1000, 1000, 1000],
            "track": [0, 0, 1, 0, 0, 1],
            "pitch": [20, 30, 40, 20, 30, 40],
            "dur": 1000,
            "velocity": 50,
        }
    )
    assert fileio.midi_to_df("test.mid").equals(
        expected
    ), "Writing to MIDI doesn't copy notes after excerpt_length"

    df.track = 2
    fileio.df_to_csv(df, "test.csv")
    fileio.csv_to_midi(
        "test.csv", "test.mid", existing_midi_path="test2.mid", excerpt_length=1000
    )
    expected = pd.DataFrame(
        {
            "onset": [0, 0, 0, 1000, 1000, 1000],
            "track": [2, 2, 2, 0, 0, 1],
            "pitch": [20, 30, 40, 20, 30, 40],
            "dur": 1000,
            "velocity": 50,
        }
    )
    assert fileio.midi_to_df("test.mid").equals(
        expected
    ), "Writing to MIDI with extra track breaks"

    csv_path = "test.csv"
    midi_path = "test.mid"
    fileio.df_to_csv(CLEAN_INPUT_DF, csv_path)
    # Some less robust tests regarding single_track and non_overlapping
    # (Robust versions will be in the *_to_df and df_to_* functions)
    for (track, overlap) in itertools.product([False, True], repeat=2):
        kwargs = {"single_track": track, "non_overlapping": overlap}

        df = fileio.csv_to_df(csv_path, **kwargs)
        fileio.df_to_midi(df, midi_path)
        correct = fileio.midi_to_df(midi_path)

        fileio.csv_to_midi(csv_path, midi_path, **kwargs)
        res = fileio.midi_to_df(midi_path)

        assert res.equals(
            correct
        ), f"csv_to_midi not using args correctly with args={kwargs}"

    for filename in ["test.mid", "test2.mid", "test.csv"]:
        try:
            os.remove(filename)
        except Exception:
            pass
