"""Quick utils for files and plotting used by the adjacent notebooks."""
import logging
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pretty_midi
import requests
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


# File handling ========================================================================
def download_file(url, local_filename=None, min_size_bytes=100):
    """Downloads file at url."""
    if local_filename is None:
        local_filename = url.rsplit("/")[-1]
    file_exists = Path(local_filename).exists()
    file_corrupt = False
    if file_exists:
        LOGGER.info("file already exists at %s, not downloading", local_filename)
        file_size = Path(local_filename).stat().st_size
        if file_size < min_size_bytes:
            LOGGER.warning(
                "file less than %d bytes, downloading again.", min_size_bytes
            )
            file_corrupt = True
    if (not file_exists) or file_corrupt:
        req_result = requests.get(url, allow_redirects=True)
        with open(f"{local_filename}", "wb") as filehandle:
            filehandle.write(req_result.content)


def unzip_file(zipfile_path):
    stem = Path(zipfile_path).stem
    if Path(stem).exists():
        LOGGER.error("%s exists. Delete and retry.", stem)
        return
    with zipfile.ZipFile(zipfile_path, "r") as zz:
        for file in tqdm(zz.namelist(), total=len(zz.namelist())):
            zz.extract(member=file, path=stem)


# Plotting =============================================================================
DEFAULT_PATCH_KWARGS = dict(
    facecolor="None",
    edgecolor="black",
    alpha=0.8,
    linewidth=1,
    capstyle="round",
    linestyle="-",
)


def plot_from_df_track(df, ax=None, pitch_spacing=0.05, patch_kwargs=None):
    """Produce a 'pianoroll' style plot from a note DataFrame"""
    if ax is None:
        ax = plt.gca()

    note_boxes = []

    for _, row in df.iterrows():
        onset, pitch, dur = row[["onset", "pitch", "dur"]]
        box_height = 1 - 2 * pitch_spacing
        box_width = dur
        x = onset
        y = pitch - (0.5 - pitch_spacing)
        bottom_left_corner = (x, y)
        rect = Rectangle(bottom_left_corner, box_width, box_height)
        note_boxes.append(rect)

    kwargs = DEFAULT_PATCH_KWARGS.copy()
    if patch_kwargs is not None:
        kwargs.update(patch_kwargs)
    pc = PatchCollection(note_boxes, **kwargs)

    ax.add_collection(pc)
    ax.autoscale()  # required for boxes to be seen
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return pc


def plot_from_df(
    df,
    ax=None,
    pitch_spacing=0.05,
    axes_labels=["onset", "pitch"],
    track_patch_kwargs=None,
    tracks="all",
    legend=False,
    **all_patch_kwargs,
):
    if ax is None:
        ax = plt.gca()

    if tracks == "all":
        tracks = df.track.unique()

    if track_patch_kwargs is None:
        track_patch_kwargs = {}
        for ii, track in enumerate(tracks):
            track_patch_kwargs[track] = {"facecolor": f"C{ii}"}
    for track in tracks:
        track_df = df[df["track"] == track]
        patch_kwargs = {**all_patch_kwargs, **track_patch_kwargs[track]}
        plot_from_df_track(
            track_df, ax=ax, pitch_spacing=pitch_spacing, patch_kwargs=patch_kwargs
        )
    legend_elements = [
        Patch(**track_patch_kwargs[track], label=f"track {track}") for track in tracks
    ]
    if legend:
        plt.legend(handles=legend_elements, loc="best")
    if axes_labels is not False:
        xlabel, ylabel = axes_labels
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("pitch")


def show_gridlines(ax=None, major_mult=4, minor_mult=1, y_maj_min=None):
    """Convenience method to apply nice major and minor gridlines to pianoroll
    plots"""
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(minor_mult))
    ax.xaxis.set_major_locator(MultipleLocator(major_mult))
    if y_maj_min is not None:
        major_mult, minor_mult = y_maj_min
        ax.yaxis.set_minor_locator(MultipleLocator(minor_mult))
        ax.yaxis.set_major_locator(MultipleLocator(major_mult))
    ax.grid(which="major", linestyle="-", axis="both")
    ax.grid(which="minor", linestyle="--", axis="both")
    ax.set_axisbelow(True)


def plot_against_orig(orig_excerpt, list_of_diff_excerpts):
    nr_diffs = len(list_of_diff_excerpts)
    fig, ax = plt.subplots(
        1, nr_diffs + 1, figsize=(6 * (nr_diffs + 1), 4), sharex=True, sharey=True
    )
    plt.sca(ax[0])
    plot_from_df(orig_excerpt, alpha=0.3)
    plt.title("original")
    for ii in range(nr_diffs):
        plt.sca(ax[ii + 1])
        plot_from_df(orig_excerpt, alpha=0.3)
        plot_from_df(list_of_diff_excerpts[ii])
        plt.title(f"deg {ii+1}")


# Synthesis ===================================================================
DEFAULT_QUANTIZATION = 12


def note_df_to_pretty_midi(note_df, inst_name="Acoustic Grand Piano"):
    """Create a pretty_midi.PrettyMIDI object from a note DataFrame

    Notes
    -----
    See http://www.midi.org/techspecs/gm1sound.php
    """
    midi = pretty_midi.PrettyMIDI()
    quantum_time = 1 / 1000
    tracks = note_df.track.unique()
    if isinstance(inst_name, str):
        inst_name = {track: inst_name for track in tracks}
    for track, track_df in note_df.groupby("track"):
        track_inst_name = inst_name[track]
        inst_program = pretty_midi.instrument_name_to_program(track_inst_name)
        inst = pretty_midi.Instrument(program=inst_program)
        for _, row in track_df.iterrows():
            onset, pitch, dur = row[["onset", "pitch", "dur"]]
            start_time = quantum_time * onset
            end_time = quantum_time * (onset + dur)
            note = pretty_midi.Note(
                velocity=100, pitch=pitch, start=start_time, end=end_time
            )
            inst.notes.append(note)
        midi.instruments.append(inst)
    return midi


def synthesize_from_note_df(note_df, fs=16000, inst_name="Acoustic Grand Piano"):
    """Create a waveform from a note DataFrame"""
    midi = note_df_to_pretty_midi(note_df, inst_name=inst_name)
    return midi.synthesize(fs=fs)
