#!/usr/bin/env python
"""Script to measure the errors from a transcription system in order to create
a degraded MIDI dataset with the given proportions of degradations."""
import argparse
import glob
import os

from mdtk import degradations


def parse_args(args_input=None):
    parser = argparse.ArgumentParser(description="Measure errors from a "
                                     "transcription error in order to make "
                                     "a degraded MIDI dataset with the measure"
                                     " proportion of each degration.")
    parser.add_argument("--gt", help="The directory which contains the ground "
                        "truth musical scores or piano rolls.")
    parser.add_argument("--trans", help="The directory which contains the "
                        "transcriptions")
    args = parser.parse_args(args=args_input)
    return args



if __name__ == '__main__':
    args = parse_args()
    
    # TODO: Decide on data type (or allow multiple types)
    trans = glob.glob(os.path.join(args.trans, '*.midi'))
    
    proportion = np.zeros((len(degradations.get_degradations()), 0))
    
    for file in trans:
        basename = os.path.splitext(os.path.basename(file))[0]
        # TODO: Decide on data type (or allow multiple types)
        gt = os.path.join(args.gt, basename + '.midi')
        
        # TODO: Also get some parameters?
        proportion = np.vstack((proportions, get_proportions(gt, trans)))
        
    proportion = np.mean(proportion, axis=0)
    
    # TODO: Write out to json file
    
    