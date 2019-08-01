"""The main data structure class we will transform into and all the functions
for converting between different data formats.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator

import pretty_midi


CSV_COLNAMES = ['onset', 'pitch', 'morph', 'dur', 'ch']
DEFAULT_QUANTIZATION = 12


def read_note_csv(path, colnames=CSV_COLNAMES):
    """Read csv as defined in mirex competition"""
    df = pd.read_csv(path, names=colnames)
    return df


def plot_flat_pianoroll(mat, fignum=None):
    """Plot with matshow - depreciated, use plot_from_pianoroll"""
    plt.matshow(mat, aspect='auto', origin='lower', fignum=fignum)
    plt.colorbar()


def show_gridlines(ax=None, major_mult=4, minor_mult=1, y_maj_min=None):
    """Convenience method to apply nice major and minor grilines to pianoroll
    plots"""
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(minor_mult))
    ax.xaxis.set_major_locator(MultipleLocator(major_mult))
    if y_maj_min is not None:
        major_mult, minor_mult = y_maj_min
        ax.yaxis.set_minor_locator(MultipleLocator(minor_mult))
        ax.yaxis.set_major_locator(MultipleLocator(major_mult))
    ax.grid(which='major', linestyle='-', axis='both')
    ax.grid(which='minor', linestyle='--', axis='both')
    ax.set_axisbelow(True)


def round_to_nearest_division(x, div):
    """Rounds an array of numbers to the nearest division of a unit.

    Parameters
    ----------
    x : array like
        array of numbers to round (or single number)
    div : int
        Number of divisions per unit

    Returns
    -------
    array like x

    Notes
    -----
    In the case of x*div producing exact .5 values, np.round rounds
    to the nearest even number.
    """
    return np.round(x*div) / div


# Conversion from note dataframe ==============================================
def make_df_monophonic(df, inplace=False):
    """Takes a note df and returns a version where all notes which overlap
    a subsequent note have their durations clipped. Assumes that the note_df
    input is sorted accordingly."""
    if not inplace:
        df = df.copy(deep=True)
    onsets = df.onset.unique()
    next_onset = {onsets[ii]: onsets[ii+1] for ii in range(len(onsets)-1)}
    note_off = df.onset + df.dur
    next_onset[onsets[-1]] = note_off.max()
    # select notes which overlap their next onset
    idx = note_off > [next_onset[oo] for oo in df.onset]
    # clip the duration of these notes
    new_durations = [next_onset[oo] - oo for oo in df.loc[idx, 'onset']]
    dtype = df.dur.dtype
    df.loc[idx, 'dur'] = new_durations
    if dtype == int:  # This is a bit of a hack to handle the assigment
                      # on the line above, which makes ints into floats.
                      # It appears this has something to do with nans, (nans
                      # are not valid for numpy ints, so pandas converts)
                      # however new_durations never contains nans, so this is 
                      # probably something to do with how .loc works...
        df.dur = df.dur.astype(dtype)
    if not inplace:
        return df


def preprocess_df_and_check_args(df, quantize=False, sort=False,
                                 monophonic=False, max_note_len=None,
                                 return_quanta_labels=False,
                                 quantization=None):
    """Convenience function to declutter df_to_pianoroll"""
    if sort:
        df = df.sort_values('onset')
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.copy()

    if max_note_len:
        df.loc[df.dur > max_note_len, 'dur'] = max_note_len

    if not quantize:
        assert all(df.dtypes[['onset', 'dur']] == int), ('onset and dur in '
            'the supplied dataframe must have type int. Maybe this df is not '
            'quantized? Supply a value for quantize to handle this.')
        quant_df = df
    else:
        if quantization:
            print(f'Warining: quantization ({quantization}) being overwritten '
                  f'with quantize ({quantize})')
        quantization = quantize
        quant_df = quantize_df(df, quantization, inplace=False)

    del df  # free memory

    if monophonic:
        quant_df = make_df_monophonic(quant_df)

    if return_quanta_labels and (quantization is None):
        raise ValueError('Must set a quantization level if returning labels')

    return quant_df, quantization


def df_to_pianoroll(df, quantize=False, sort=False, monophonic=False,
                    max_note_len=None, return_quanta_labels=False,
                    quantization=None):
    """Create a pianoroll matrix from a list of note events. Expects a
    pandas dataframe containing columns labelled 'onset', 'pitch', and 'dur'.
    Handles polyphonic sequences. Output is a 2 x 129 dim vector per
    quantized time step - the vector dimensions represent the 128 midinote
    values + 1 silence token, the first vector describes if pitches are
    sounding at the beginning of the quantum, the second vector describes if a
    pitch ends at the end of the quantum.

    If the duration of a note would overlap the onset of the next, the note
    duration is cut short such that it ends before the next begins.

    The input df is assumed sorted by onset time and indexed from 0. If not,
    set sort to True.

    Parameters
    ----------
    df : DataFrame
        A pandas dataframe in the supplied csv format. Assumed to have columns
        labelled onset, pitch, and dur for the onset time, midinote pitch
        number and the duration (in crotchets).
    quantize : int
        If specified, quantization will be performed on the incoming DataFrame.
        The integer value of quantize is the number of quanta per crotchet -
        this will overwrite quantization if specified.
        If False (the default), assumes the onsets are already integer quanta.
        Will error if onsets and durations are not integers.
        quantization
    monophonic : bool
        Enforce the piece to be monophonic. Defaults to False. Will throw a
        warning if there are multiple pitches with the same onset time -
        the piece was never made as monophonic - but will select the lower
        pitch.
    sort : bool
        Whether to sort the supplied dataframe by onset time and reindex
    max_note_len : float (optional)
        Maximum length a note can be in number of crotchets/quanta. This
        truncation is performed first, so this number relates to the input
        dataframe duration unit.
    return_quanta_labels : bool
        Whether to return labels for the quanta. The labels are the original
        onset times before quantization.
    quantization : int (optional)
        The number of quanta per crotchet. Only required if wanting to return
        the labels for the quanta

    Outputs
    -------
    pianoroll : np.array size (128, T, 2)

    """
    quant_df, quantization = preprocess_df_and_check_args(
            df, quantize=quantize, sort=sort, monophonic=monophonic,
            max_note_len=max_note_len, quantization=quantization,
            return_quanta_labels=return_quanta_labels
    )

    first_onset = quant_df.onset.iloc[0]
    last_offset = (quant_df.onset + quant_df.dur).max()
    nr_quanta = last_offset - first_onset
    pianoroll = np.zeros((129, nr_quanta, 2), dtype=int)

    # zero the onsets for easy indexing of output df
    quant_df.onset = quant_df.onset - first_onset
    # This is a single instrument so pitches cannot overlap
    # N.B. df.groupby preserves the sorted order within each grouped df
    for pitch, pitch_df in quant_df.groupby('pitch'):
        for iloc in range(len(pitch_df)):
            idx = pitch_df.index[iloc]
            onset = pitch_df.at[idx, 'onset']
            dur = pitch_df.at[idx, 'dur']
            trunc_note_dur = False
            if iloc != len(pitch_df)-1:
                next_idx = pitch_df.index[iloc+1]
                next_onset = pitch_df.at[next_idx, 'onset']
                if onset + dur > next_onset:
                    trunc_note_dur = True
            if trunc_note_dur:
                offset = next_onset - 1
            else:
                offset = onset + dur - 1
            pianoroll[pitch, onset:(offset+1), 0] = 1
            pianoroll[pitch, offset, 1] = 1  # the offset dimension
    del quant_df  # free memory

    # Silence token: makes the top matrix one-hot in the monophonic case
    pianoroll[-1, np.sum(pianoroll[:, :, 0], axis=0) == 0, 0] = 1
    # Give the silence note-off events - this is for plotting ease
    for ii in range(pianoroll.shape[1]):
        if ii < (pianoroll.shape[1] - 1):
            if pianoroll[-1, ii, 0] == 1 and pianoroll[-1, ii+1, 0] == 0:
                pianoroll[-1, ii, 1] = 1
        else:
            assert pianoroll[-1, ii, 0] != 1, ('Something went wrong: piece '
                'should never end with silence token')

    if return_quanta_labels:
        quanta_labels = [onset_quantum / quantization for onset_quantum in
                         range(first_onset, last_offset+1)]
        return quanta_labels, pianoroll
    return pianoroll


# Quantization ================================================================
def quantize_df(df, quantization, monophonic=False, inplace=False):
    """Quantization is number of divisions per crotchet. Will return onsets and
    durations as an integer number of quanta.

    Parameters
    ----------
    df : DataFrame
        A pandas dataframe in the supplied csv format. Assumed to have columns
        labelled onset, pitch, and dur for the onset time, midinote pitch
        number and the duration (in crotchets).
    quantization : int
        Number of divisions requred per integer (i.e. quarter note or crotchet)
    monophonic : bool (optional)
        If True, enforces the output to be monophonic.
    inplace : bool (optional)
        If True, do operation inplace on supplied DataFrame and return None

    Returns
    -------
    this_df : DataFrame
        The quantized DataFrame

    Notes
    -----
    If the result of the onset or duration multiplied by the quantization
    results in an exact .5 value, np.around will round to the nearest even
    number.
    
    Whilst the monophonic argument may seem superfluous if the input df is
    monophonic, note that the quantisation could create polyphony. This
    argument allows the user to handle this.
    """
    if inplace:
        this_df = df
    else:
        this_df = df.copy()
    this_df.onset = np.around(this_df.onset * quantization).astype(int)
    this_df.dur = np.around(this_df.dur * quantization).astype(int)
    this_df.loc[this_df['dur']==0, 'dur'] = 1  # minimum duration is 1 quantum
    if monophonic:
        this_df = make_df_monophonic(this_df)
    if not inplace:
        return this_df


# Pianoroll ===================================================================
def get_pianoroll_onsets(pianoroll):
    """Return a 2D binary matrix indicating note onset locations"""
    if pianoroll.ndim == 3:
        pr_prev = np.zeros_like((pianoroll))
        # the previous values in time at a given time
        pr_prev[:, 1:, :] = pianoroll[:, :-1, :]
        initial_onsets = ((pianoroll[:, :, 0] - pr_prev[:, :, 0]) == 1)
        post_offset_onsets = ((pianoroll[:, :, 0] + pr_prev[:, :, 1]) == 2)
        return initial_onsets.astype(int) + post_offset_onsets.astype(int)
    elif pianoroll.ndim == 2:
        pr_prev = np.zeros_like((pianoroll))
        # the previous values in time at a given time
        pr_prev[:, 1:] = pianoroll[:, :-1]
        initial_onsets = ((pianoroll - pr_prev) == 1)
        return initial_onsets.astype(int)
    else:
        raise ValueError('pianoroll must have 2 or 3 dims not '
                         f'{pianoroll.ndim}')


def get_pianoroll_noteoffs(pianoroll):
    """Return a 2D binary matrix indicating note off locations"""
    if pianoroll.ndim == 3:
        return pianoroll[:, :, 1]
    elif pianoroll.ndim == 2:
        pr_next = np.zeros_like((pianoroll))
        # the next values in time at a given time
        pr_next[:, :-1] = pianoroll[:, 1:]
        note_off = ((pianoroll - pr_next) == 1)
        return note_off.astype(int)
    else:
        raise ValueError('pianoroll must have 2 or 3 dims not '
                         f'{pianoroll.ndim}')


def pianoroll_to_df(pianoroll, exclude_silence_token=True,
                    first_onset=None, quantization=None):
    """Return a DataFrame object describing the note events in a pianoroll

    Parameters
    ----------
    pianoroll : np.array
        A 2 or 3 dimensional binary array pianoroll
    exclude_silence_token : bool (optional)
        If False, do not exclude any pitch dimensions. If left true, assume
        pitch at location -1 is the silence token and exclude it.
    first_onset : float
        A value to add to all the onsets in the resulting DataFrame

    Returns
    -------
    df : DataFrame
        A pandas dataframe representing the note events
    """
    if exclude_silence_token:
        # TODO: this should error if size was 128 already
        # TODO: change this when dims changes
        if pianoroll.shape[0] == 128:
            raise ValueError('ERROR: trying to remove a silence token, '
                             'but dim 0 is size 128...')
        pianoroll = pianoroll[:-1]
    onset_mat = get_pianoroll_onsets(pianoroll)
    offset_mat = get_pianoroll_noteoffs(pianoroll)
    pitch, onset = np.where(onset_mat)
    _, offset = np.where(offset_mat)
    # We can do this because we know the order is fixed and that every onset
    # has an associated offset
    df = pd.DataFrame({'onset': onset, 'pitch': pitch, 'dur': offset-onset+1})
    df.sort_values(['onset', 'pitch'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if quantization:
        df.onset = df.onset / quantization
        df.dur = df.dur / quantization
    if first_onset:
        df.onset = df.onset + first_onset
    return df


# Plotting ====================================================================
def plot_from_df(df, ax=None, facecolor='None', edgecolor='black', alpha=.8,
                 linewidth=1, capstyle='round', pitch_spacing=0.05,
                 linestyle='-', label_axes=True):
    """Produce a 'pianoroll' style plot from a note DataFrame"""
    if ax is None:
        ax = plt.gca()

    note_boxes = []

    for _, row in df.iterrows():
        onset, pitch, dur = row[['onset', 'pitch', 'dur']]
        box_height = 1-2*pitch_spacing
        box_width = dur
        x = onset
        y = pitch - (0.5-pitch_spacing)
        bottom_left_corner = (x, y)
        rect = Rectangle(bottom_left_corner, box_width, box_height)
        note_boxes.append(rect)

    pc = PatchCollection(note_boxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor, capstyle=capstyle,
                         linewidth=linewidth, linestyle=linestyle)

    ax.add_collection(pc)

    ax.autoscale()  # required for boxes to be seen
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if label_axes:
        ax.set_xlabel('onset')
        ax.set_ylabel('pitch')
    return pc


def plot_from_pianoroll(pianoroll, exclude_silence_token=True,
                        label_axes=False, first_onset=None,
                        **kwargs):
    """Produce a 'pianoroll' style plot from a 2D or 3D pianoroll matrix"""
    df = pianoroll_to_df(
        pianoroll,
        exclude_silence_token=exclude_silence_token,
        first_onset=first_onset
    )
    return plot_from_df(df, label_axes=label_axes, **kwargs)


# Synthesis ===================================================================
def note_df_to_pretty_midi(note_df, bpm=100, inst_name='Acoustic Grand Piano'):
    """Create a pretty_midi.PrettyMIDI object from a note DataFrame

    Notes
    -----
    See http://www.midi.org/techspecs/gm1sound.php
    """
    midi = pretty_midi.PrettyMIDI()
    inst_program = pretty_midi.instrument_name_to_program(inst_name)
    inst = pretty_midi.Instrument(program=inst_program)
    quantum_time = 60 / bpm
    for _, row in note_df.iterrows():
        onset, pitch, dur = row[['onset', 'pitch', 'dur']]
        start_time = quantum_time * onset
        end_time = quantum_time * (onset+dur)
        note = pretty_midi.Note(velocity=100, pitch=pitch,
                                start=start_time, end=end_time)
        inst.notes.append(note)
    midi.instruments.append(inst)
    return midi


def synthesize_from_quant_df(quant_df, bpm=100, fs=16000,
                             inst_name='Acoustic Grand Piano',
                             quantization=12):
    note_df = quant_df.copy()
    note_df.onset = note_df.onset / quantization
    note_df.dur = note_df.dur / quantization
    return synthesize_from_note_df(note_df, bpm, fs, inst_name)
    


def synthesize_from_note_df(note_df, bpm=100, fs=16000,
                            inst_name='Acoustic Grand Piano'):
    """Create a waveform from a note DataFrame"""
    midi = note_df_to_pretty_midi(note_df, bpm=bpm, inst_name=inst_name)
    return midi.synthesize(fs=fs)


def synthesize_from_pianoroll(
        pianoroll,
        quantization=DEFAULT_QUANTIZATION,
        bpm=100,
        fs=16000,
        inst_name='Acoustic Grand Piano',
        exclude_silence_token=True):
    """Create a waveform from a 2D or 3D pianoroll object"""
    quant_df = pianoroll_to_df(pianoroll,
                               exclude_silence_token=exclude_silence_token)
    note_df = quant_df
    note_df.onset = note_df.onset / quantization
    note_df.dur = note_df.dur / quantization
    return synthesize_from_note_df(note_df, bpm=bpm, fs=fs,
                                   inst_name=inst_name)


# Class wrapper ===============================================================
class Composition:
    """
    Wrapper class for note csv data. Has methods to plot, synthesize,
    quantize, and convert this data between required formats e.g. pianoroll.
    There also methods to check the data and reformat it into the
    required format e.g. sorting by onset then pitch.

    The input note_df should be a pandas DataFrame with columns called
    onset, pitch, and dur.

    Additionally there are class methods for import from other formats e.g.
    directly from csv files with the expected columns.
    """
    default_quantization = DEFAULT_QUANTIZATION

    def __init__(self, note_df=None, csv_path=None, sort_note_df=False,
                 check_sorted=True, quantization=default_quantization,
                 monophonic=False, max_note_len=None):

        self.quantization = quantization
        self.monophonic = monophonic
        
        if csv_path and (note_df is not None):
            raise ValueError('Supply either csv_path or note_df, not both')
        elif csv_path:
            self.csv_path = csv_path
            self.note_df = read_note_csv(csv_path)
        elif note_df is not None:
            self.csv_path = None
            self.note_df = note_df
        else:
            raise ValueError('Supply at least one of csv_path or note_df')

        if sort_note_df:
            self.note_df = self.note_df.sort_values(['onset', 'pitch'])
            self.note_df.reset_index(drop=True, inplace=True)
        elif check_sorted:
            # check monotonic increasing onset,pitch and index is a range
            assert isinstance(self.note_df.index, pd.RangeIndex), ("Index of "
                "note_df must be a pd.RangeIndex. Set sort_note_df = True to "
                "handle this automatically.")
            assert (
                self.note_df.equals(
                    self.note_df
                        .sort_values(['onset', 'pitch'])
                        .reset_index(drop=True)
                )
            ), ("note_df must be sorted by onset then pitch. Set sort_note_df "
                "= True to handle this automatically.")
        
        if monophonic:
            make_df_monophonic(self.note_df, inplace=True)
        
        if max_note_len:
            idx = self.note_df.dur > max_note_len
            self.note_df.loc[idx, 'dur'] = max_note_len
        
        # Properties - only created upon first get
        self._quant_df = None
        self._pianoroll = None
        self._quanta_labels = None


    def __repr__(self):
        """String representation of the composition - the note_df"""
        return self.note_df.__repr__()


    def __str__(self):
        """String representation of the composition - the note_df"""
        return self.note_df.__str__()


    # Properties ==============================================================
    @property
    def quant_df(self):
        """Only create upon first get. No set method - will error if set."""
        if self._quant_df is None:
            self._quant_df = quantize_df(self.note_df, self.quantization,
                                         monophonic=self.monophonic, 
                                         inplace=False)
        return self._quant_df

    @property
    def pianoroll(self):
        """Only create upon first get. No set method - will error if set."""
        if self._pianoroll is None:
            self._quanta_labels, self._pianoroll = df_to_pianoroll(
                self.quant_df,
                quantize=False,  # already quantized
                return_quanta_labels=True,
                quantization=self.quantization
            )
        return self._pianoroll

    @property
    def quanta_labels(self):
        """Only create upon first get. No set method - will error if set."""
        if self._quanta_labels is None:
            first_onset = self.quant_df.onset.iloc[0]
            last_noteoff = (self.quant_df.onset + self.quant_df.dur).max()
            self._quanta_labels = [quantum / self.quantization for quantum
                                   in range(first_onset, last_noteoff+1)]
        return self._quanta_labels


    # Construction ============================================================
    @classmethod
    def from_note_df(cls, note_df, quantization=default_quantization,
                     **kwargs):
        """Method for instantiation from a pre-made note_df. This isn't really
        needed...kept in case things change in future"""
        return cls(note_df=note_df, quantization=quantization, **kwargs)

    @classmethod
    def from_csv(cls, csv_path, quantization=default_quantization, **kwargs):
        """Method for instantiation from a csv file. This isn't really
        needed...kept in case things change in future"""
        return cls(csv_path=csv_path, quantization=quantization, **kwargs)

    @classmethod
    def from_quant_df(cls, quant_df, quantization=default_quantization,
                      **kwargs):
        """Method for instantiation from a pre-made quantized note_df."""
        note_df = quant_df.copy()
        note_df.onset = note_df.onset / quantization
        note_df.dur = note_df.dur / quantization
        c = cls(note_df=note_df, quantization=quantization, **kwargs)
        c._quant_df = quant_df  # set directly to avoid re-computation
        return c

    @classmethod
    def from_quant_pianoroll(cls, quant_pianoroll,
                             quantization=default_quantization, **kwargs):
        """Method for instantiation from a pre-made quantized pianoroll."""
        quant_df = pianoroll_to_df(quant_pianoroll)
        c = cls.from_quant_df(quant_df, quantization=quantization, **kwargs)
        # set directly to avoid re-computation
        c._quant_df = quant_df
        c._pianoroll = quant_pianoroll
        return c


    # Plotting ================================================================
    def plot(self, quantized=True, label_axes=True, **kwargs):
        """Convenience method for plotting either a quantized or note level
        pianoroll. By default will label the axes to clarify what type of
        plot is shown. Plot kwargs for plot_from_df can be supplied.

        Parameters
        ----------
        quantized : bool
            If False, the labels for the x axis (time) are given in the
            original 'note level' units, else counts number of quanta.
        label_axes : bool
            Whether or not label the x and y axes

        Returns
        -------
        None
        """
        if quantized:
            plot_from_df(self.quant_df, label_axes=label_axes, **kwargs)
            show_gridlines(major_mult=self.quantization*4,
                           minor_mult=self.quantization)
            if label_axes:
                plt.xlabel('onset quantum')
        else:
            plot_from_df(self.note_df, label_axes=label_axes, **kwargs)
            show_gridlines()


    # Synthesis ===============================================================
    def synthesize(self, bpm=100, fs=16000, inst_name='Acoustic Grand Piano',
                   trim=True):
        """Creates a waveform which can be listened to. Makes a conversion to
        a pretty-midi MIDI object behind the scences.

        Parameters
        ----------
        bpm : int
            Beats per minute - the speed the composition should be synthesized
            at
        fs : int
            The sampling freqency - the number of waveform values per second
        inst_name : string
            The name of the midi instrument to be used for synthesis. See
            http://www.midi.org/techspecs/gm1sound.php for a valid list
        """
        if trim:
            note_df = self.note_df.copy()
            note_df.onset = note_df.onset - note_df.onset.iloc[0]
        else:
            note_df = self.note_df
        return synthesize_from_note_df(note_df, bpm=bpm, fs=fs,
                                       inst_name=inst_name)
