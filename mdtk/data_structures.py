"""The main data structure class we will transform into and all the functions
for converting between different data formats.
"""
import copy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator

import pretty_midi



NOTE_DF_SORT_ORDER = ['onset', 'track', 'pitch', 'dur']
DEFAULT_QUANTIZATION = 12
NR_MIDINOTES = 128



def read_note_csv(path, onset='onset', pitch='pitch', dur='dur', track='ch',
                  sort=True, header='infer', make_monophonic=None,
                  max_note_len=None):
    """Read a csv and create a standard note event DataFrame - a `note_df`.
    
    Parameters
    ----------
    path : str
        The path to the csv to be imported
    onset : str or int
        The name or index of the column in the csv describing note onset times
    pitch : str or int
        The name or index of the column in the csv describing note pitches
    dur : str or int
        The name or index of the column in the csv describing note durations
    track : str or int
        The name or index of the column in the csv describing the midi track
    sort : bool
        Whether to sort the resulting DataFrame to the default sort order
    header : int, list of int, default ‘infer’
        parameter to pass to pandas read_csv - see their documentation. Must
        set to None if your csv has no header.
    make_monophonic : list(int), 'all', True, or None
        Track names to make monophonic. If None, performs no action. If True,
        expects a single track, and makes it monophonic. If 'all', makes all
        tracks monophonic.
    max_note_len : int, float, or None
        A value for the maximum duration of a note
    """
    cols = {}
    cols[onset] = 'onset'
    cols[pitch] = 'pitch'
    cols[dur] = 'dur'
    if track is not None:
        cols[track] = 'track'
    string_colspec = all(isinstance(vv, int) for vv in cols.keys())
    integer_colspec = all(isinstance(vv, str) for vv in cols.keys())
    assert string_colspec or integer_colspec, ('All column specifications '
        'must be either all strings or all integers.')
    
    df = pd.read_csv(path, header=header, usecols=list(cols.keys()))
    df.rename(columns=cols, inplace=True)
    if track is None:
        df.loc[:, 'track'] = 0
    
    # Check no overlapping notes of the same pitch
    df = df.groupby(['track', 'pitch']).apply(fix_overlapping_notes)
    
    if make_monophonic is not None:
        if make_monophonic is True:
            make_monophonic = df.track.unique()
            assert len(make_monophonic) == 1, ('Expected only one track, found'
                f' {len(make_monophonic)}: {make_monophonic}'
            )
        elif make_monophonic == 'all':
            make_monophonic = df.track.unique()
        elif isinstance(make_monophonic, list):
            assert all(track_name in df.track.unique()
                       for track_name in make_monophonic), ('Not all track '
                'names supplied exist in the dataframe')
            
        for track_name in make_monophonic:
            df.loc[df['track']==track_name] = fix_overlapping_notes(
                df[df['track']==track_name].copy()  # reqd to avoid warnings
            )
            
    if max_note_len is not None:
        df.loc[df['dur'] > max_note_len] = max_note_len
        
    if sort:
        df.sort_values(by=NOTE_DF_SORT_ORDER, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df.loc[:, NOTE_DF_SORT_ORDER]


def fix_overlapping_notes(df, drop_new_cols=True):
    """For use in a groupby operation over track. Fixes any pitches that
    overlap. Pulls the offending note's offtime back to one quantum behind the
    following onset time."""
    if df.shape[0] <= 1:
        return df
    df['note_off'] = df['onset'] + df['dur']
    # need to use df.index to avoid 'chaining' iloc and loc, see:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
    df['next_note_on'] = df['onset'].shift(-1)
    df.loc[df.index[-1], 'next_note_on'] = np.inf
    df['bad_note'] = df['note_off'] > df['next_note_on']  # equal is fine
    df.loc[df['bad_note'], 'dur'] = (df.loc[df['bad_note'], 'next_note_on']
                                     - df.loc[df['bad_note'], 'onset'])
    if drop_new_cols:
        new_cols = ['note_off', 'next_note_on', 'bad_note']
        df.drop(new_cols, axis=1, inplace=True)
    return df


# Note DataFrame checking======================================================
# Most subsequent functions expect a note_df - a pandas dataframe with specific
# columns, and a defined sort order. We want to write functions which are
# usable independent of the Composition class, but also by the Composition
# class. The Composition class does checking on the note_df by default, but
# users who call these functions may not. Subsequent functions all call this
# checking by default but, when used in the Composition class, they don't need
# to
def check_note_df(note_df):
    """Performs checks to ensure the note_df has the correct properties"""
    # Check columns
    assert all(col in note_df.columns for col in NOTE_DF_SORT_ORDER), (
              f'note_df must contain all columns in {NOTE_DF_SORT_ORDER}')
    # Check has increasing integer index
    assert all(note_df.index == pd.RangeIndex(note_df.shape[0])), ('note_df '
        'must have a RangeIndex with integer steps')
    # Check sorted
    assert (
        note_df
            .sort_values(by=NOTE_DF_SORT_ORDER)
            .reset_index(drop=True)
            .equals(note_df)
        ), (f'note_df must be sorted by {NOTE_DF_SORT_ORDER} and columns '
            'ordered')
    # Check no overlapping notes of the same pitch
    for track, track_df in note_df.groupby('track'):
        for pitch, pitch_df in track_df.groupby('pitch'):
            note_off = pitch_df.onset + pitch_df.dur
            next_note_on = pitch_df.onset.shift(-1)
            next_note_on.iloc[-1] = np.inf
            try:
                note_off <= next_note_on
            except ValueError:
                print(note_off)
                print(next_note_on)
            assert all(note_off <= next_note_on), (f'Track {track} '
                f'has an overlapping note at pitch {pitch}, i.e. there is a '
                'note which overlaps another at the same pitch')


def check_monophonic(note_df):
    assert len(note_df.track.unique()), ('This note_df is not monophonic, it '
        'has multiple tracks')
    note_off = note_df.onset + note_df.dur
    next_note_on = note_df.onset.iloc[1:]
    assert all(note_off.iloc[:-1] > next_note_on), (f'There is a  note which '
        'overlaps another')
    

# note_df_editing =============================================================
# Functions for altering already imported note_df DataFrames
def make_df_monophonic(df, inplace=False):
    """Takes a note df and returns a version where all notes which overlap
    a subsequent note have their durations clipped. Assumes that the note_df
    input is sorted accordingly."""
    # TODO: account for tracks
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


# Quantization ================================================================
def quantize_df(df, quantization, inplace=False):
    """Quantization is number of divisions per integer. Will return onsets and
    durations as an integer number of quanta.

    Parameters
    ----------
    df : DataFrame
        A pandas dataframe in the supplied csv format. Assumed to have columns
        labelled onset, pitch, and dur for the onset time, midinote pitch
        number and the duration (in crotchets).
    quantization : int
        Number of divisions requred per integer (i.e. quarter note or crotchet)
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
    if not inplace:
        return this_df


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
    ax.grid(which='major', linestyle='-', axis='both')
    ax.grid(which='minor', linestyle='--', axis='both')
    ax.set_axisbelow(True)


def plot_matrix(mat, fignum=None):
    """Plot a 2d matrix with matshow. Used for debugging."""
    plt.matshow(mat, aspect='auto', origin='lower', fignum=fignum)
    plt.colorbar()


# Synthesis ===================================================================
# TODO: make all these work with tracks
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
                             quantization=DEFAULT_QUANTIZATION):
    note_df = quant_df.copy()
    note_df.onset = note_df.onset / quantization
    note_df.dur = note_df.dur / quantization
    return synthesize_from_note_df(note_df, bpm, fs, inst_name)
    


def synthesize_from_note_df(note_df, bpm=100, fs=16000,
                            inst_name='Acoustic Grand Piano'):
    """Create a waveform from a note DataFrame"""
    midi = note_df_to_pretty_midi(note_df, bpm=bpm, inst_name=inst_name)
    return midi.synthesize(fs=fs)



# Class wrapper ===============================================================
class Pianoroll():
    """
    Wrapper class for arrays representing a pianoroll of some description. It
    is callable and will return an array. It only stores the matrices, and is
    initialised with a standard note dataframe or a pianoroll array
    
    It expects to recieve a dataframe with integer onset times i.e. you have
    already quantized the data.
    
    Parameters
    ----------
    
    """
    def __init__(self, quant_df=None, pianoroll_array=None, note_off=False,
                 first_note_on=None, quantization=None):
        # Properties only created upon first get but stored thereafter
        self._note_off = None
        self._pianoroll = None
        self._shape = None
        # -----
        
        self.first_note_on = None
        if (quant_df is not None) and (pianoroll_array is not None):
            raise ValueError('Supply either quant_df or pianoroll_array, '
                             'not both')
        elif quant_df is not None:
            expected_cols = ['onset', 'pitch', 'dur']
            assert all([col in quant_df.columns for col in expected_cols]), (
                f'quant_df is expected to have columns {expected_cols}')
            assert all(pd.api.types.is_integer_dtype(tt)
                       for tt in quant_df[['onset', 'pitch', 'dur']].dtypes), (
                'quant_df must have integer data (the data are expected to '
                'have been quantized)')
            quant_df_note_off = quant_df.onset + quant_df.dur
            self.first_note_on = quant_df.onset.min()
            self.nr_timesteps = quant_df_note_off.max() - self.first_note_on
            try:
                self.track_names = quant_df.track.unique()
                self.nr_tracks = len(self.track_names)
            except AttributeError:
                quant_df = quant_df.copy()
                quant_df['track'] = 1
                self.track_names = np.array([1])
                self.nr_tracks = 1
            self.sounding, self.note_on = self.get_sounding_note_on(
                    df=quant_df
                )
            if note_off is True:
                self._note_off = self.note_off(df=quant_df)
        elif pianoroll_array is not None:
            assert pianoroll_array.ndim == 4, ('pianoroll_array must be a 4 '
                'dimensional array with dimensions representing: '
                '(track, channel, pitch, time)')
            if pianoroll_array.shape[1] == 1:
                # This is a flat pianoroll with just 'sounding' channel
                self.sounding = pianoroll_array.squeeze()
                self.note_on = self.get_note_on_from_sounding()
            elif pianoroll_array.shape[1] == 2:
                # This is a pianoroll with channels 'sounding', 'note_on'
                self.sounding = pianoroll_array[:, 0, :, :]
                self.note_on = pianoroll_array[:, 1, :, :]
            else:
                raise NotImplementedError()
            self.nr_timesteps = self.sounding.shape[-1]
            self.nr_tracks = pianoroll_array.shape[0]
            self.track_names = np.arange(self.nr_tracks)
        else:
            raise ValueError('Supply at least one of quant_df or '
                             'pianoroll_array')
        
        if self.first_note_on is None:
            self.first_note_on = first_note_on
        elif first_note_on is not None:  # specifying a note_on time overrides
            self.first_note_on = first_note_on
        
        self.quantization = quantization
    
    
    def __call__(self, channels=['sounding', 'note_on'], tracks='all'):
        """Upon call, the object will present as a numpy array.
            
        Parameters
        ----------
        channels: list
            An ordered list of what information to put in each channel
            dimension. Names of channels requested must match attributes
            available to the class.
        tracks: list or 'all'
            Which instrument tracks to include
        
        Returns
        -------
        pianoroll: np.array
            In general, the output array will be of shape:
            (nr_tracks, nr_channels, nr_pitches, nr_timesteps) 
        """
        if tracks == 'all':
            tracks = np.arange(self.nr_tracks)
        else:
            assert all([track in self.track_names for track in tracks]), (
                f'a track in {tracks} is not in {self.track_names}'
            )
            tracks = [np.argmax(self.track_names == track) for track in tracks]
        pianoroll = np.zeros((len(tracks), len(channels), NR_MIDINOTES,
                              self.nr_timesteps), dtype='uint8')
        for ii, attr_name in enumerate(channels):
            array = getattr(self, attr_name)
            pianoroll[:, ii, :, :] = array
        return pianoroll
    
    
    def __repr__(self):
        """String representation of the composition - the note_df"""
        return self.pianoroll.__repr__()


    def __str__(self):
        """String representation of the composition - the note_df"""
        return self.pianoroll.__str__()
    
    
    def __getitem__(self, key):
        return self.pianoroll.__getitem__(key)
    
    
    def __len__(self):
        return self.nr_timesteps


    def __eq__(self, other):
        return self.pianoroll == other
    
    
    # Properties ==============================================================
    @property
    def note_off(self, df=None):
        """Only create upon first get. No set method - will error if set."""
        if self._note_off is None:
            if df is None:
                self._note_off = self.get_note_off_from_pianoroll()
            else:
                self._note_off = self.get_note_off(df)
        return self._note_off
    
    @property
    def pianoroll(self):
        """Only create upon first get. No set method - will error if set."""
        if self._pianoroll is None:
            self._pianoroll = self()
        return self._pianoroll
    
    @property
    def shape(self):
        """Only create upon first get. No set method - will error if set."""
        return self.pianoroll.shape
    
    # Note_df methods =========================================================
    def get_sounding_note_on(self, df):
        df_ = df.set_index('track', append=True).reorder_levels([1, 0])
        sounding = np.zeros((self.nr_tracks, NR_MIDINOTES, self.nr_timesteps),
                            dtype='uint8')
        note_on = np.zeros((self.nr_tracks, NR_MIDINOTES, self.nr_timesteps),
                           dtype='uint8')
        for ii, track in enumerate(self.track_names):
            for idx, row in df_.loc[track].iterrows():
                note_on[ii, row.pitch, row.onset] = 1
                note_off = row.onset + row.dur
                sounding[ii, row.pitch, row.onset:note_off] = 1
        return sounding, note_on
    
    
    def get_note_off(self, df):
        df_ = df.set_index('track', append=True).reorder_levels([1, 0])
        note_off = np.zeros((self.nr_tracks, NR_MIDINOTES, self.nr_timesteps),
                           dtype='uint8')
        for ii, track in enumerate(self.track_names):
            for idx, row in df_.loc[track].iterrows():
                note_off[ii, row.pitch, row.onset+row.dur] = 1
        return note_off
    
    
    # Internal methods ========================================================
    def get_note_off_from_pianoroll(self):
        note_off = np.zeros_like(self.sounding, dtype='uint8')
        sounding_back_1 = np.zeros_like(self.sounding, dtype='uint8')
        sounding_back_1[:, :, :-1] = self.sounding[:, :, 1:]
        note_off[np.logical_and(self.sounding == 1,
                                sounding_back_1 == 0)] = 1
        note_on_back_1 = np.zeros_like(self.note_on, dtype='uint8')
        note_on_back_1[:, :, :-1] = self.note_on[:, :, 1:]
        note_off[np.logical_and(self.sounding == 1,
                                note_on_back_1 == 1)] = 1
        return note_off
    
    
    def get_note_on_from_sounding(self):
        note_on = np.zeros_like(self.sounding, dtype='uint8')
        sounding_fwd_1 = np.zeros_like(self.sounding, dtype='uint8')
        sounding_fwd_1[:, :, 1:] = self.sounding[:, :, :-1]
        note_on[np.logical_and(self.sounding == 1, sounding_fwd_1 == 0)] = 1
        return note_on
    
   
    def get_note_df(self, first_note_on=None, quantization=None):
        """Return a DataFrame object describing the note events in a pianoroll
    
        Parameters
        ----------
        first_note_on : float
            A value to add to all the onsets in the resulting DataFrame
        quantization : int
            The number of divisions per integer there are in the pianoroll
    
        Returns
        -------
        df : DataFrame
            A pandas dataframe representing the note events
        """
        if first_note_on is None:
            first_note_on = self.first_note_on
        if quantization is None:
            quantization = self.quantization
        
        track, pitch, note_on = np.where(self.note_on)
        _, _, note_off = np.where(self.note_off)
        # We can do this because we know the order is fixed and that every
        # note_on has an associated note_off
        df = pd.DataFrame({'onset': note_on, 'track': track, 'pitch': pitch,
                           'dur': note_off - note_on + 1})
        df['track'] = df['track'].apply(
                lambda track_nr: self.track_names[track_nr]
            )
        df.sort_values(by=NOTE_DF_SORT_ORDER, inplace=True)
        df.reset_index(drop=True, inplace=True)
        check_note_df(df)
        
        if quantization:
            df.onset = df.onset / quantization
            df.dur = df.dur / quantization
        if first_note_on:
            df.onset = df.onset + first_note_on
        return df
    
    
    # Plotting ================================================================
    def plot(self, first_note_on=None, quantization=None,
             **plot_from_df_kwargs):
        df = self.get_note_df(first_note_on=first_note_on,
                              quantization=quantization)
        ax = plot_from_df(df, **plot_from_df_kwargs)
        return ax
    
    
    # Synthesis ===============================================================
    def synthesize(
            self,
            quantization=DEFAULT_QUANTIZATION,
            bpm=100,
            fs=16000,
            inst_name='Acoustic Grand Piano'
        ):
        """Create a waveform array"""
        df = self.get_note_df(first_note_on=0, quantization=quantization)
        return synthesize_from_note_df(df, bpm=bpm, fs=fs, inst_name=inst_name)
    
    
    # Functional ==============================================================
    def copy(self):
        """Returns a deep copy of the object."""
        return copy.deepcopy(self)
    
    

class Composition:
    """Wrapper class for note csv data. Takes as input either and already
    imported pandas DataFrame, or the location of a csv to be imported. For the
    former case, checks are done to ensure this is a valid `note_df`. In the
    latter case, this is handled automatically.
    
    The definition of a valid `note_df`:
        * has columns ['onset', 'track', 'pitch', 'dur'] which describe
            * onset - the time a note begins
            * track - which 'instrument' is playing the note
            * pitch - the midinote pitch number the note is sounding
            * dur   - the duration of the note
        * is sorted by ['onset', 'track', 'pitch', 'dur']
        * has an increasing integer index
    
    Anything involving pianorolls will require a quanitization of the
    `note_df`. The default quantization is 12 divisions per integer note onset,
    which is good if the onset column describes the number of quarter notes
    (crotchets), but if the onset column describes seconds, or miliseconds,
    this may require adjustment.
    
    The Composition class has methods to plot, synthesize, quantize, and
    convert this data between required formats e.g. pianoroll, note streams,
    etc.

    Additionally there are class methods for import from other formats e.g.
    directly from csv files with the expected columns.
    
    Parameters
    ----------
    note_df : pd.DataFrame
        The DataFrame to be wrapped. Must have required columns, see the
        description above. Either this or the csv_path must be specified
    csv_path : str
        Path to a csv to import. Either this or the note_df must be specified.
        If this parameter is given, you must specify parameters for the import,
        `read_note_csv_kwargs` if they differ from the default. See the docs
        for `read_note_csv` for more information
    read_note_csv_kwargs : dict
        These kwargs are supplied to `read_note_csv` if a csv is imported.
        These include:
        onset : str or int
            The name or index of the column in the csv describing note onset
            times
        pitch : str or int
            The name or index of the column in the csv describing note pitches
        dur : str or int
            The name or index of the column in the csv describing note
            durations
        track : str or int
            The name or index of the column in the csv describing the midi
            track
        sort : bool
            Whether to sort the resulting DataFrame to the default sort order
        header : int, list of int, default ‘infer’
            parameter to pass to pandas read_csv - see their documentation.
            Must set to None if your csv has no header.
    quantization : int
        A quantization level to use. This is the number of divisions to make
        per integer increase in the time columns: `onset` and `dur`. A
        quantized version of the note data must be used in order to create a
        pianoroll. By default, this is set at 12: 12 divisions per quarter note
        (crotched) is useful for encoding triplets, but for data which are not
        already beat aligned, one may like to use a higher quantization per
        second or milisecond.
    make_monophonic : list(int), 'all', True, or None
        Track names to make monophonic. If None, performs no action. If
        True, expects a single track, and makes it monophonic. If 'all',
        makes all tracks monophonic. Note that this overrides the arg passed
        to read_note_csv in read_note_csv_kwargs
    max_note_len : int, float, or None
        A value for the maximum duration of a note. Note that this overrides
        the arg passed to read_note_csv in read_note_csv_kwargs
    """
    default_quantization = DEFAULT_QUANTIZATION
    default_read_note_csv_kwargs = dict(
        onset='onset',
        pitch='pitch',
        dur='dur',
        track='ch',
        sort=True,
        header='infer'
    )
    def __init__(self, note_df=None, csv_path=None,
                 read_note_csv_kwargs=default_read_note_csv_kwargs,
                 quantization=default_quantization, make_monophonic=None,
                 max_note_len=None):
        self.quantization = quantization
        self.make_monophonic = make_monophonic
        self.max_note_len = max_note_len
        
        # Create note_df from either the path or supplied df
        if csv_path and (note_df is not None):  # Don't supply both!
            raise ValueError('Supply either csv_path or note_df, not both')
        elif csv_path:
            self.csv_path = csv_path
            read_note_csv_kwargs['make_monophonic'] = self.make_monophonic
            read_note_csv_kwargs['max_note_len'] = self.max_note_len
            self.read_note_csv_kwargs = read_note_csv_kwargs
            self.note_df = read_note_csv(csv_path, **read_note_csv_kwargs)
        elif note_df is not None:
            self.csv_path = None
            self.read_note_csv_kwargs = None
            self.note_df = note_df
            # We do not assume that the supplied note_df is correctly formed,
            # and simply bomb out if it is not
            # TODO: implement df methods to fix issues instead e.g. overlaps.
            #       Copy code from read_note_csv. e.g.:
            #       * reorder columns
            #       * if all columns but track and no extra cols, assume 1 trk
            if self.make_monophonic is not None:
                # TODO: adapt `make_df_monophonic` to handle tracks
                raise NotImplementedError()
                make_df_monophonic(self.note_df, inplace=True)
            if self.max_note_len is not None:
                idx = self.note_df.dur > max_note_len
                self.note_df.loc[idx, 'dur'] = max_note_len
            try:
                check_note_df(note_df)
            except AssertionError as e:
                print('The note_df supplied fails to meet the requirements. '
                      'Please reformat it such that it passes `check_note_df` '
                      'or supply a csv path instead.')
                raise e
        else:
            raise ValueError('Supply at least one of csv_path or note_df')
        
        self.track_names = note_df.track.unique()
        self.nr_tracks = len(self.track_names)
        
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
                                         inplace=False)
        return self._quant_df

    @property
    def pianoroll(self):
        """Only create upon first get. No set method - will error if set."""
        if self._pianoroll is None:
            self._pianoroll = Pianoroll(self.quant_df)
        return self._pianoroll

    @property
    def quanta_labels(self):
        """Only create upon first get. No set method - will error if set.
        Quanta labels are the labels for the quanta in self.quant_df in the
        original units (self.note_df_time_unit)"""
        if self._quanta_labels is None:
            first_note_on = self.quant_df.onset.iloc[0]
            last_note_off = (self.quant_df.onset + self.quant_df.dur).max()
            self._quanta_labels = [quantum / self.quantization for quantum
                                   in range(first_note_on, last_note_off+1)]
        return self._quanta_labels


    # Construction ============================================================
    @classmethod
    def from_note_df(cls, note_df, **kwargs):
        """Method for instantiation from a pre-made note_df. This isn't really
        needed...kept in case things change in future"""
        return cls(note_df=note_df, **kwargs)

    @classmethod
    def from_csv(cls, csv_path, **kwargs):
        """Method for instantiation from a csv file. This isn't really
        needed...kept in case things change in future"""
        return cls(csv_path=csv_path, **kwargs)

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
    def from_pianoroll_array(cls, pianoroll_array, first_note_on=None, 
                             quantization=None, **kwargs):
        """Method for instantiation from an array representing a pianoroll. The
        array is expected to be of shape (tracks, channels, pitches, timesteps)
        and channels must be size 1 or 2, if 1 the pianoroll just shows
        'sounding' pitches, and if 2 it shows 'sounding' and 'note_on' events.
        """
        raise NotImplementedError()
        pianoroll = Pianoroll(pianoroll_array=pianoroll_array)
        note_df = pianoroll.get_note_df(
            first_note_on=first_note_on,
            quantization=quantization
        )
        c = cls(note_df=note_df, quantization=quantization, **kwargs)
        # set directly to avoid re-computation
        c._pianoroll = pianoroll
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


    # Functional ==============================================================
    def copy(self):
        """Returns a deep copy of the object."""
        return copy.deepcopy(self)
