"""Tools to convert from the default acme csvs and metadata.csv to specific
formats easy for the provided pytorch DataLoaders"""
import logging
import os

import numpy as np
import pandas as pd
import tqdm

from mdtk.degradations import MAX_PITCH_DEFAULT, MIN_PITCH_DEFAULT
from mdtk.df_utils import NOTE_DF_SORT_ORDER
from mdtk.fileio import DEFAULT_VELOCITY, csv_to_df


# Convenience function...
def diff_pd(df1, df2):
    """Identify differences between two pandas DataFrames"""
    assert (df1.columns == df2.columns).all(), "DataFrame column names are different"
    if any(df1.dtypes != df2.dtypes):
        "Data Types are different, trying to convert"
        df2 = df2.astype(df1.dtypes)
    if df1.equals(df2):
        return None
    else:
        # need to account for np.nan != np.nan returning True
        diff_mask = (df1 != df2) & ~(df1.isnull() & df2.isnull())
        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]
        changed.index.names = ["id", "col"]
        difference_locations = np.where(diff_mask)
        changed_from = df1.values[difference_locations]
        changed_to = df2.values[difference_locations]
        return pd.DataFrame(
            {"from": changed_from, "to": changed_to}, index=changed.index
        )


class CommandVocab(object):
    def __init__(
        self,
        min_pitch=MIN_PITCH_DEFAULT,
        max_pitch=MAX_PITCH_DEFAULT,
        time_increment=40,
        max_time_shift=4000,
        specials=["<pad>", "<unk>", "<eos>", "<sos>"],
    ):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        # itos - integer to string
        self.itos = (
            list(specials)
            + [f"o{ii}" for ii in range(min_pitch, max_pitch + 1)]  # special tokens
            + [f"f{ii}" for ii in range(min_pitch, max_pitch + 1)]  # note_on
            + [  # note_off
                f"t{ii}"
                for ii in range(time_increment, max_time_shift + 1, time_increment)
            ]
        )  # time_shift
        self.stoi = {tok: ii for ii, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


def create_corpus_csvs(acme_dir, format_dict):
    """
    From a given acme dataset, create formatted csv files to use with
    our provided pytorch Dataset classes.

    Parameters
    ----------
    acme_dir : string
        The directory containing the acme data.

    format_dict: dict
        A dictionary (likely one provided in FORMATTERS), containing at least:
        name : string
            The name to print in the loading message.
        prefix : string
            The string to prepend to "_corpus_path" and "_corpus_lin_nr" columns
            in the resulting metadata.csv file, as well as to use in the names
            of the resulting corpus-specific csv files like:
            {split}_{prefix}_corpus.csv
        df_to_str : function
            The function to convert from a pandas DataFrame to a string in the
            desired format.
    """
    name = format_dict["name"]
    prefix = format_dict["prefix"]
    df_converter_func = format_dict["df_to_str"]
    fh_dict = {
        split: open(os.path.join(acme_dir, f"{split}_{prefix}_corpus.csv"), "w")
        for split in ["train", "valid", "test"]
    }
    line_counts = {split: 0 for split in ["train", "valid", "test"]}
    meta_df = pd.read_csv(os.path.join(acme_dir, "metadata.csv"))
    for idx, row in tqdm.tqdm(
        meta_df.iterrows(), total=meta_df.shape[0], desc=f"Creating {name} corpus"
    ):
        alt_df = csv_to_df(os.path.join(acme_dir, row.altered_csv_path))
        alt_str = df_converter_func(alt_df)
        clean_df = csv_to_df(os.path.join(acme_dir, row.clean_csv_path))
        clean_str = df_converter_func(clean_df)
        deg_num = row.degradation_id
        split = row.split
        fh = fh_dict[split]
        fh.write(f"{alt_str},{clean_str},{deg_num}\n")
        meta_df.loc[idx, f"{prefix}_corpus_path"] = os.path.basename(fh.name)
        meta_df.loc[idx, f"{prefix}_corpus_line_nr"] = line_counts[split]
        line_counts[split] += 1
    meta_df.loc[:, f"{prefix}_corpus_line_nr"] = meta_df[
        f"{prefix}_corpus_line_nr"
    ].astype(int)
    meta_df.to_csv(os.path.join(acme_dir, "metadata.csv"), index=False)


def df_to_pianoroll_str(df, time_increment=40):
    """
    Convert a given pandas DataFrame into a packed piano-roll representation:
    Each string will look like:

    "notes1_onsets1/notes2_onsets2/..."

    where notes and onsets are space-separated strings of pitches. notes
    contains those pitches which are present at each frame, and onsets
    contains those pitches which have and onset at a given frame.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame which we will convert into the piano-roll.

    time_increment : int
        The length of a single frame, in milliseconds.
    """
    # Input validation
    assert time_increment > 0, "time_increment must be positive."

    quant_df = df.loc[:, ["pitch"]]
    quant_df["onset"] = (df["onset"] / time_increment).round().astype(int)
    quant_df["offset"] = (
        ((df["onset"] + df["dur"]) / time_increment)
        .round()
        .astype(int)
        .clip(lower=quant_df["onset"] + 1)
    )

    # Create piano rolls
    length = quant_df["offset"].max()
    max_pitch = quant_df["pitch"].max() + 1
    note_pr = np.zeros((length, max_pitch))
    onset_pr = np.zeros((length, max_pitch))
    for _, note in quant_df.iterrows():
        onset_pr[note.onset, note.pitch] = 1
        note_pr[note.onset : note.offset, note.pitch] = 1

    # Pack into format
    strings = []
    for note_frame, onset_frame in zip(note_pr, onset_pr):
        strings.append(
            " ".join(map(str, np.where(note_frame == 1)[0]))
            + "_"
            + " ".join(map(str, np.where(onset_frame == 1)[0]))
        )

    return "/".join(strings)


def pianoroll_str_to_df(pr_str, time_increment=40):
    """
    Convert a given piano roll string into a pianoroll

    Parameters
    ----------
    pr_str : string
        The pianoroll string, created by df_to_pianoroll_str.

    Returns
    -------
    df : pd.DataFrame
        A dataframe equal to the given pianoroll string.
    """
    notes = []
    active = [-1] * 128

    frames = pr_str.split("/")

    for frame_num, frame in enumerate(frames):
        time = frame_num * time_increment
        note_pitches, onset_pitches = frame.split("_")
        if note_pitches != "":
            note_pitches = list(map(int, note_pitches.split(" ")))
        else:
            note_pitches = []

        # Check that all pitches continue
        for pitch, idx in enumerate(active):
            if idx >= 0 and pitch not in note_pitches:
                # Pitch doesn't continue
                notes[idx]["dur"] = time - notes[idx]["onset"]
                active[pitch] = -1

        if onset_pitches == "":
            continue

        # Check onsets for new notes/breaks in existing notes
        for pitch in map(int, onset_pitches.split(" ")):
            if active[pitch] >= 0:
                # Pitch was active. Stop it here.
                notes[active[pitch]]["dur"] = time - notes[active[pitch]]["onset"]

            # Start new note
            active[pitch] = len(notes)
            notes.append(
                {
                    "onset": time,
                    "pitch": pitch,
                    "track": 0,
                    "dur": None,
                    "velocity": DEFAULT_VELOCITY,
                }
            )

    # Close any still open notes
    for idx in active:
        if idx >= 0:
            notes[idx]["dur"] = len(frames) * time_increment - notes[idx]["onset"]

    # Create df
    df = pd.DataFrame(notes)
    df = df.sort_values(by=NOTE_DF_SORT_ORDER)[NOTE_DF_SORT_ORDER].reset_index(
        drop=True
    )
    return df


def double_pianoroll_to_df(
    pianoroll,
    min_pitch=MIN_PITCH_DEFAULT,
    max_pitch=MAX_PITCH_DEFAULT,
    time_increment=40,
):
    """
    Convert a double pianoroll (sustain and onset, as output by a task 4 model),
    into a DataFrame for use in evaluation.

    Parameters
    ----------
    pianoroll : np.ndarray
        A pianoroll of shape (n, 2 * max_pitch - min_pitch + 1), where n is the
        number of frames, the left half of the matrix is the sustain_pr, and the
        right half is the onset_pr.

    min_pitch : int
        The pitch at pianoroll indices [:, 0] and [:, max_pitch - min_pitch + 1].

    max_pitch : int
        The pitch at pianoroll indices [:, max_pitch - min_pitch] and [:, -1].

    time_increment : int
        The length of a single frame, in milliseconds.

    Returns
    -------
    df : pd.DataFrame
        A dataframe equal to the given pianoroll.
    """

    if max_pitch != pianoroll.shape[1] / 2 + min_pitch - 1:
        logging.warning(
            "max_pitch doesn't match pianoroll shape and min_pitch. "
            "Setting max_pitch to "
            f"{int(pianoroll.shape[1] / 2 + min_pitch - 1)}."
        )
        max_pitch = int(pianoroll.shape[1] / 2 + min_pitch - 1)

    df_notes = []
    active = [-1] * (max_pitch - min_pitch + 1)  # Index of active note in notes
    midpoint = int(pianoroll.shape[1] / 2)

    for frame_num, (notes, onsets) in enumerate(
        zip(pianoroll[:, :midpoint], pianoroll[:, midpoint:])
    ):
        time = frame_num * time_increment
        note_pitches = np.where(notes == 1)[0]
        onset_pitches = np.where(onsets == 1)[0]

        # Check that all pitches continue
        for pitch, idx in enumerate(active):
            if idx >= 0 and pitch not in note_pitches:
                # Pitch doesn't continue
                df_notes[idx]["dur"] = time - df_notes[idx]["onset"]
                active[pitch] = -1

        # Check onsets for new notes/breaks in existing notes
        for pitch in onset_pitches:
            if active[pitch] >= 0:
                # Pitch was active. Stop it here.
                df_notes[active[pitch]]["dur"] = time - df_notes[active[pitch]]["onset"]

            # Start new note
            active[pitch] = len(df_notes)
            df_notes.append(
                {"onset": time, "pitch": pitch + min_pitch, "track": 0, "dur": None}
            )

        # Find pitch presences that should've been onsets but weren't
        for pitch in note_pitches:
            if active[pitch] < 0:
                # Pitch is supposed to be inactive, but there is a sustain
                # Treat this as an onset
                active[pitch] = len(df_notes)
                df_notes.append(
                    {"onset": time, "pitch": pitch + min_pitch, "track": 0, "dur": None}
                )

    # Close any still open notes
    frames = pianoroll.shape[0]
    for idx in active:
        if idx >= 0:
            df_notes[idx]["dur"] = frames * time_increment - df_notes[idx]["onset"]

    # Create df
    if len(df_notes) > 0:
        df = pd.DataFrame(df_notes)
        df = df.sort_values(by=NOTE_DF_SORT_ORDER)[NOTE_DF_SORT_ORDER].reset_index(
            drop=True
        )
    else:
        df = pd.DataFrame(columns=NOTE_DF_SORT_ORDER).reset_index(drop=True)
    return df


def df_to_command_str(
    df,
    min_pitch=MIN_PITCH_DEFAULT,
    max_pitch=MAX_PITCH_DEFAULT,
    time_increment=40,
    max_time_shift=4000,
):
    """
    Convert a given pandas DataFrame into a sequence commands, note_on (o),
    note_off (f), and time_shift (t). Each command is followed by a number:
    o10 means note_on<midinote 10>, t60 means time_shift<60 ms>. It is assumed
    df has been sorted by onset.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame which we will convert into commands.

    min_pitch : int
        The minimum pitch at which notes will occur.

    max_pitch : int
        The maximum pitch at which notes will occur.

    time_increment : int
        The length of a single frame, in milliseconds.

    max_time_shift : int
        The maximum shift length, in milliseconds. Must be divisible by
        time_increment.

    Returns
    -------
    command_string : str
        The string containing a space separated list of commands.
    """
    # Input validation
    assert (
        max_time_shift % time_increment == 0
    ), "max_time_shift must be divisible by time_increment."
    assert max_pitch >= min_pitch, "max_pitch must be >= min_pitch."
    assert time_increment > 0, "time_increment must be positive."
    assert max_time_shift > 0, "max_time_shift must be positive."

    note_off = df.loc[:, ["onset", "pitch"]]
    note_off["onset"] = note_off["onset"] + df["dur"]
    note_off["cmd"] = note_off["pitch"].apply(lambda x: f"f{x}")
    note_off["cmd_type"] = "f"
    note_on = df.loc[:, ["onset", "pitch"]]
    note_on["cmd"] = note_off["pitch"].apply(lambda x: f"o{x}")
    note_on["cmd_type"] = "o"
    commands = pd.concat((note_on, note_off))
    commands["onset"] = (commands["onset"] / time_increment).round().astype(
        int
    ) * time_increment
    commands = commands.sort_values(
        ["onset", "cmd_type", "pitch"], ascending=[True, True, True]
    )

    command_list = []
    current_onset = commands.onset.iloc[0]
    for idx, row in commands.iterrows():
        while current_onset != row.onset:
            time_shift = min(row.onset - current_onset, max_time_shift)
            command_list += [f"t{time_shift}"]
            current_onset += time_shift
        command_list += [f"{row.cmd}"]

    return " ".join(command_list)


def command_str_to_df(cmd_str):
    """
    Convert a given string of commands back to a pandas DataFrame.

    Parameters
    ----------
    cmd_str : str
        The string containing a space separated list of commands.

    Returns
    -------
    df : pd.DataFrame
        The pandas DataFrame representing the note data.
    """
    commands = cmd_str.split()
    note_on_pitch = []
    note_on_time = []
    note_off_pitch = []
    note_off_time = []
    curr_time = 0
    for cmd_str in commands:
        cmd = cmd_str[0]
        value = int(cmd_str[1:])
        if cmd == "o":
            note_on_pitch += [value]
            note_on_time += [curr_time]
        elif cmd == "f":
            note_off_pitch += [value]
            note_off_time += [curr_time]
        elif cmd == "t":
            curr_time += value
        else:
            raise ValueError(f"Invalid command {cmd}")
    df = pd.DataFrame(columns=NOTE_DF_SORT_ORDER, dtype=int)
    for ii, (pitch, onset) in enumerate(zip(note_on_pitch, note_on_time)):
        note_off_idx = note_off_pitch.index(pitch)  # gets first instance
        note_off_pitch.pop(note_off_idx)
        off = note_off_time.pop(note_off_idx)
        dur = off - onset
        track = 0
        velocity = DEFAULT_VELOCITY
        df.loc[ii] = [onset, track, pitch, dur, velocity]

    return df


FORMATTERS = {
    "command": {
        "name": "command",
        "prefix": "cmd",
        "df_to_str": df_to_command_str,
        "str_to_df": command_str_to_df,
        "model_to_df": None,
        "message": (
            "The {train,valid,test}_cmd_corpus.csv are command-based "
            "(note_on, note_off, shift) versions of the acme data more "
            "convenient for our provided pytorch Dataset classes."
        ),
        "deg_label": "deg_cmd",
        "clean_label": "clean_cmd",
        "task_labels": ["deg_label", "deg_label", None, "clean_cmd"],
        "dataset": "CommandDataset",
        "models": [
            "Command_ErrorDetectionNet",
            "Command_ErrorClassificationNet",
            None,  # Probably won't implement. Ground truth is not created.
            None,
        ],
    },
    "pianoroll": {
        "name": "pianoroll",
        "prefix": "pr",
        "df_to_str": df_to_pianoroll_str,
        "str_to_df": pianoroll_str_to_df,
        "model_to_df": double_pianoroll_to_df,
        "message": (
            "The {train,valid,test}_pr_corpus.csv are piano-roll-based "
            "versions of the acme data more convenient for our provided "
            "pytorch Dataset classes."
        ),
        "deg_label": "deg_pr",
        "clean_label": "clean_pr",
        "task_labels": ["deg_label", "deg_label", "changed_frames", "clean_pr"],
        "dataset": "PianorollDataset",
        "models": [
            None,
            None,
            "Pianoroll_ErrorLocationNet",
            "Pianoroll_ErrorCorrectionNet",
        ],
    },
}
