#!/usr/bin/env python
"""Script to generate Altered and Corrupted Midi Exerpt (ACME) datasets"""
import sys
import json
import argparse

from mdtk import degradations, downloaders, data_structures, midi

# For dev mode warnings...
if not sys.warnoptions:
    import os
    import warnings
    warnings.simplefilter("always") # Change the filter in this process
    os.environ["PYTHONWARNINGS"] = "always" # Also affect subprocesses



with open('./img/logo.txt', 'r') as ff:
    LOGO = ff.read()
DEFAULT_DATASETS = ['PPDDSept2018Monophonic']
DESCRIPTION = "Make datasets of altered and corrupted midi exerpts."



def parse_degradation_kwargs(kwarg_dict):
    """Convenience function to parse a dictionary of keyword arguments for
    the functions within the degradations module. All keys in the kwarg_dict
    supplied should start with the function name and be followed by a double
    underscore.

    Parameters
    ----------
    kwarg_dict : dict
        Dict containing keyword arguments to parse

    Returns
    -------
    func_kwargs : dict
        Dict with keys matching the names of the functions. The corresponding
        value is a dictionary of the keyword arguments for the function.
    """
    func_names = degradations.DEGRADATIONS.keys()
    func_kwargs = {name: {} for name in func_names}
    if kwarg_dict is None:
        return func_kwargs
    for kk, kwarg_value in kwarg_dict.items():
        try:
            func_name, kwarg_name = kk.split('__', 1)
        except ValueError:
            raise ValueError(f"Supplied keyword [{kk}] must have a double "
                              "underscore")
        assert func_name in func_names, f"{func_name} not in {func_names}"
        func_kwargs[func_name][kwarg_name] = kwarg_value
    return func_kwargs



def parse_args(args_input=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-o', '--output-dir', type=str, default='./acme',
                        help='the directory to write the dataset to (defaults '
                        'to ./acme)')
    parser.add_argument('-i', '--input-dir', type=str, default='./input_data',
                        help='the directory to store the preprocessed '
                        'downloaded data to (by default, makes a directory '
                        'input_data in the current directory')
    parser.add_argument('--local-midi-dirs', metavar='midi_dir', type=str,
                        nargs='*', help='directories containing midi files to '
                        'include in the dataset')
    parser.add_argument('--local-csv-dirs', metavar='csv_dir', type=str,
                        nargs='*', help='directories containing csv files to '
                        'include in the dataset')
    parser.add_argument('--datasets', metavar='dataset_name',
                        nargs='*', choices=downloaders.DATASETS,
                        help='datasets to download and use. Must match names '
                        'of classes in the downloaders module. By default, '
                        'will use cached downloaded data if available, see '
                        '--download-cache-dir and --clear-download-cache',
                        )
    parser.add_argument('--download-cache-dir', type=str,
                        default=downloaders.DEFAULT_CACHE_PATH, help='The '
                        'directory to use for storing intermediate downloaded '
                        'data e.g. zip files, and prior to preprocessing. By '
                        f'default is set to {downloaders.DEFAULT_CACHE_PATH}')
    parser.add_argument('--clear-download-cache', action='store_true',
                        help='clear downloaded data cache')
    parser.add_argument('--degradations', metavar='deg_name', nargs='*',
                        choices=list(degradations.DEGRADATIONS.values()),
                        help='degradations to use on the data. Must match '
                        'names of functions in the degradations module. By '
                        'default, will use them all.')
    parser.add_argument('--degradation-kwargs', metavar='json_string',
                        help='json with keyword arguments for the '
                        'degradation functions. First provide the degradation '
                        'name, then a double underscore, then the keyword '
                        'argument name, followed by the value to use for the '
                        'kwarg. e.g. {"pitch_shift__distribution": "poisson", '
                        '"pitch_shift__min_pitch: 5"}',
                        type=json.loads, default=None)
    parser.add_argument('--degradation-kwarg-json', metavar='json_file',
                        help='A file containing parameters as described in '
                        '--degradation-kwargs', type=json.load, default=None)
    parser.add_argument('--degradation-dist', metavar='relative_probability',
                        nargs='*', default=None, help='a list of relative '
                        'probabilities that each degradation will used. Must '
                        'be the same length as --degradations')
    args = parser.parse_args(args=args_input)
    return args


if __name__ == '__main__':
    print(LOGO)
    args = parse_args()
    print(args)
    assert (args.degradation_kwargs is None or 
            args.degradation_kwarg_json is None), ("Don't specify both "
        "--degradation-kwargs and --degradation-kwarg-json")
    if args.degradation_kwarg_json is None:
        degradation_kwargs = parse_degradation_kwargs(args.degradation_kwargs)
    else:
        degradation_kwargs = parse_degradation_kwargs(
                args.degradation_kwarg_json
            )
    # TODO: warn user they specified kwargs for degradation not being used
    # TODO: bomb out if degradation-dist is a diff length to degradations
    
    # Set up directories ======================================================
    
    # Download data ===========================================================
    
    # Create all Composition objects and write clean data to output ===========
    # output to output_dir/clean/dataset_name/filename.csv
    # The reason for this is we know there will be no filename duplicates
    
    # Perform degradations and write degraded data to output ==================
    # Because we have already written clean data, we can do this in place
    # output to output_dir/degraded/dataset_name/filename.csv
    # The reason for this is filename duplicates (as above) and easy matching
    # of source and target data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    