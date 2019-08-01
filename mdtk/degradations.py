"""Code to perform the degradations i.e. edits to the midi data"""
from mdtk.data_structures import Composition


def get_degradations():
    """
    Get a dict mapping the name of each degradation to its function pointer.
    
    Returns
    =======
    degradations : dict(string -> func)
        A dict containing the name of each degradation mapped to a pointer to
        the method which performs that degradation.
    """
    return {'pitch_shift': pitch_shift,
            'time_shift': time_shift,
            'onset_shift': onset_shift,
            'offset_shift': offset_shift,
            'remove_note': remove_note,
            'add_note': add_note}


def pitch_shift(excerpt, params):



def time_shift(excerpt, params):



def onset_shift(excerpt, params):



def offset_shift(excerpt, params):



def remove_note(excerpt, params):



def add_note(excerpt, params):


