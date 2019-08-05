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
    # Check sorted and has increasing integer index
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
            next_note_on = pitch_df.onset.iloc[1:]
            assert all(note_off.iloc[:-1] > next_note_on), (f'Track {track} '
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
# Functions for altering already imported note_df
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
class Pianoroll():
    """
    Wrapper class for arrays representing a pianoroll of some description. It
    is callable and will return an array. It only stores the matrices, and is
    initialised with a standard note dataframe.
    
    It expects to recieve a dataframe with integer onset times i.e. you have
    already quantized the data.
    """
    #TODO: create test that checks if this alters quant_df supplied
    def __init__(self, quant_df, silence=False, note_off=False):
        expected_cols = ['onset', 'pitch', 'dur']
        assert all([col in quant_df.columns for col in expected_cols]), (
                f'quant_df is expected to have columns {expected_cols}')
        assert all([pd.api.types.is_integer_dtype(tt)
                    for tt in quant_df[['onset', 'pitch', 'dur']].dtypes]), (
            'quant_df must have integer data (the data are expected to have '
            'been quantized)')
        
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
        self.sounding, self.note_on = self.get_sounding_note_on(df=quant_df)
        # Properties only created upon first get
        self._note_off = None
        self._silence = None
        if note_off is True:
            self._note_off = self.note_off(df=quant_df)
        if silence is True:
            self._silence = self.silence(df=quant_df)
    
    
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
        return pianoroll.squeeze()

    
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
        return sounding.squeeze(), note_on.squeeze()
    
    
    def get_note_off(self, df):
        df_ = df.set_index('track', append=True).reorder_levels([1, 0])
        note_off = np.zeros((self.nr_tracks, NR_MIDINOTES, self.nr_timesteps),
                           dtype='uint8')
        for ii, track in enumerate(self.track_names):
            for idx, row in df_.loc[track].iterrows():
                note_off[ii, row.pitch, row.onset+row.dur] = 1
        return note_off.squeeze()
    
    
    def get_note_off_from_pianoroll(self):
        note_on_back_1 = np.zeros_like(self.note_on, dtype='uint8')
        from_slice = tuple(slice(None) for ii in 
                           range(self.note_on.ndim - 1)) + (slice(-1),)
        to_slice = (tuple(slice(None) for ii in 
                          range(self.note_on.ndim - 1))
                   + (slice(1, self.nr_timesteps),))
        note_on_back_1[from_slice] = self.note_on[to_slice]
        sounding_back_1 = np.zeros_like(self.sounding, dtype='uint8')
        sounding_back_1[from_slice] = self.sounding[to_slice]
        note_off = np.zeros_like(self.sounding, dtype='uint8')
        note_off[np.logical_and(self.sounding == 1, sounding_back_1 == 0)] = 1
        note_off[np.logical_and(self.sounding == 1, note_on_back_1 == 1)] = 1
        return note_off
    
    
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
    def silence(self, df=None):
        """Only create upon first get. No set method - will error if set."""
        if self._silence is None:
            if df is None:
                pass
            else:
                pass
        return self._silence
    
    
    
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
            # TODO: make this a pianoroll object
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
