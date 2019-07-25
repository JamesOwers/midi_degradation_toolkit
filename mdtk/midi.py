"""Code to interact with MIDI files, including parsing and converting them
to csvs."""

import pandas as pd

import pretty_midi





def midi_to_csv(midi_path, csv_path):
    """
    Convert a MIDI file into a csv file.
    
    Parameters
    ----------
    midi_path : string
        The filename of the MIDI file to parse.
        
    csv_path : string
        The filename of the csv to write out to.
    """
    df_to_csv(midi_to_df(midi_path), csv_path)
    
    
    
    
def midi_to_df(midi_path):
    """
    Get the data from a MIDI file and load it into a pandas DataFrame.
    
    Parameters
    ----------
    midi_path : string
        The filename of the MIDI file to parse.
        
    Returns
    -------
    df : DataFrame
        A pandas DataFrame containing the notes parsed from the given MIDI
        file. There will be 4 columns:
            onset: Onset time of the note, in milliseconds.
            track: The track number of the instrument the note is from.
            pitch: The MIDI pitch number for the note.
            dur: The duration of the note (offset - onset), in milliseconds.
        Sorting will be first by onset, then track, then pitch, then duration.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    notes = []
    for index, instrument in enumerate(midi.instruments):
        for note in instrument.notes:
            notes.append({'onset': note.start * 1000,
                          'track': index,
                          'pitch': note.pitch,
                          'dur': (note.end - note.start) * 1000})
            
    df = pd.DataFrame(notes)
    df = df.sort_values(['onset', 'track', 'pitch', 'dur'])
    return df.reset_index(drop=True)



def df_to_csv(df, csv_path):
    """
    Print the data from the given pandas DataFrame into a csv in the correct
    format.
    
    Parameters
    ----------
    df : DataFrame
        The DataFrame to write out to a csv. It should be in the format returned
        by midi_to_df(midi_path), with 4 columns:
            onset: Onset time of the note, in milliseconds
            track: The track number of the instrument the note is from.
            pitch: The MIDI pitch number for the note.
            dur: The duration of the note (offset - onset), in milliseconds.
        
    csv_path : string
        The filename of the csv to which to print the data. No header or index
        will be printed, and the rows will be printed in the current order of the
        DataFrame.
    """
    df.to_csv(csv_path, index=None, header=False)