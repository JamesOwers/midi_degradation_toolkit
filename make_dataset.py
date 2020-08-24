#!/usr/bin/env python
"""Script to generate Altered and Corrupted Midi Excerpt (ACME) datasets"""
import os
import sys
import json
import argparse
import warnings
import pandas as pd
import shutil
from glob import glob
from zipfile import BadZipfile

import numpy as np
from tqdm import tqdm

from mdtk import degradations, downloaders, fileio
from mdtk.filesystem_utils import make_directory
from mdtk.formatters import create_corpus_csvs, FORMATTERS

def print_warn_msg_only(message, category, filename, lineno, file=None,
                        line=None):
    print(message, file=sys.stderr)

warnings.showwarning = print_warn_msg_only


with open(os.path.join('img', 'logo.txt'), 'r') as ff:
    LOGO = ff.read()


DESCRIPTION = "Make datasets of altered and corrupted midi excerpts."


def parse_degradation_kwargs(kwarg_dict):
    """Convenience function to parse a dictionary of keyword arguments for
    the functions within the degradations module. All keys in the kwarg_dict
    supplied should start with the function name and be followed by a double
    underscore. For example: func_name__kwarg_name. An example supplied
    dictionary:
        {
            'func1__kwarg1': 7,
            'func2__kwarg1': "hello",
            'func2__kwarg2': "world"
        }
    This results in the returned dictionary:
        {
            'func1': {'kwarg1': 7},
            'func2': {'kwarg1': "hello", 'kwarg2': "world}
        }

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


def get_random_excerpt(note_df, min_notes=10, excerpt_length=5000,
                       first_onset_range=(0, 200), iterations=10):
    """
    Take a random excerpt from the given note_df, using np.random. The excerpt
    is created as follows:

    1. Pick a note at random from the input df, excluding the last `min_notes`
       notes.
    2. Take all notes which onset within `excerpt_length` ms of that note.
    3. If the excerpt does not contain at least `min_notes` notes, repeat steps
       1 and 2 until you have drawn `iterations` invalid excerpts. In that
       case, return None.
    4. If you have a valid excerpt, shift its notes so that the first note's
       onset is at time 0, then add a random number within `first_onset_range`
       to each onset.
    5. Return the resulting excerpt.

    Parameters
    ----------
    note_df : pd.DataFrame
        The input note_df, from which we want a random excerpt.

    min_notes : int
        The minimum number of notes that must be contained in a valid excerpt.

    excerpt_length : int
        The length of the resulting excerpt, in ms. All notes which onset
        within this amount of time after a randomly chosen note will be
        included in the returned excerpt.

    first_onset_range : tuple(int, int)
        The range from which to draw a random number to add to the first note's
        onset (in ms), rather than having the chosen excerpt begin at time 0.

    iterations : int
        How many times to try to obtain a valid excerpt before giving up and
        returning None.

    Returns
    -------
    excerpt : pd.DataFrame
        A random excerpt from the given note_df. None if no valid excerpt was
        found within `iterations` attempts.
    """
    if len(note_df) < min_notes:
        return None

    for _ in range(iterations):
        note_index = np.random.choice(list(note_df.index.values)
                                      [:-min_notes])
        first_onset = note_df.loc[note_index]['onset']
        excerpt = pd.DataFrame(
            note_df.loc[note_df['onset'].between(
                first_onset, first_onset + excerpt_length)])

        # Check for validity of excerpt
        if len(excerpt) < min_notes:
            excerpt = None
        else:
            break

    if excerpt is None:
        return None

    onset_shift = np.random.randint(first_onset_range[0], first_onset_range[1])
    excerpt['onset'] += onset_shift - first_onset
    excerpt = excerpt.reset_index(drop=True)
    return excerpt


def clean_download_cache(dir_path=downloaders.DEFAULT_CACHE_PATH, prompt=True):
    """
    Delete the download cache directory and all files in it.

    Parameters
    ----------
    dir_path : str
        The path to the download cache base directory.

    prompt : bool
        Prompt the user before deleting.

    Returns
    -------
    clean_ok : bool
        True if the cache has been cleaned and deleted successfully, the cache
        dir doesn't exist, or the user chooses not to delete it at the prompt.
        False otherwise.
    """
    if not os.path.exists(dir_path):
        print(f"Download cache ({dir_path}) is clear and deleted already.")
        return True

    if prompt:
        response = input(f"Delete download cache ({dir_path})? [y/N]: ")

    if (not prompt) or response in ['y', 'ye', 'yes']:
        try:
            shutil.rmtree(dir_path)
        except:
            print("Could not delete download cache. Please do so manually.")
            return False
    return True


def parse_args(args_input=None):
    """Convenience function for parsing user supplied command line args"""
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    default_outdir = os.path.join(os.getcwd(), 'acme')
    parser.add_argument('-o', '--output-dir', type=str, default=default_outdir,
                        help='the directory to write the dataset to.')
    default_indir = os.path.join(os.getcwd(), 'input_data')
    parser.add_argument('--config', default=None, help='Load a json config '
                        'file, in the format created by measure_errors.py. '
                        'This will override --degradations, --degradation-'
                        'dist, and --clean-prop.')
    parser.add_argument('--formats', metavar='format', help='Create '
                        'custom versions of the acme data for easier loading '
                        'with our provided pytorch Dataset classes. Choices are'
                        f' {list(FORMATTERS.keys())}. Specify none to avoid '
                        'creation', nargs='*', default=list(FORMATTERS.keys()),
                        choices=FORMATTERS.keys())
    parser.add_argument('--local-midi-dirs', metavar='midi_dir', type=str,
                        nargs='*', help='directories containing midi files to '
                        'include in the dataset', default=[])
    parser.add_argument('--local-csv-dirs', metavar='csv_dir', type=str,
                        nargs='*', help='directories containing csv files to '
                        'include in the dataset', default=[])
    parser.add_argument('--recursive', action='store_true', help='Search local'
                        ' dataset directories recursively for all midi or csv '
                        'files.')
    parser.add_argument('--datasets', metavar='dataset_name',
                        nargs='*', default=downloaders.DATASETS,
                        help='datasets to download and use. Must match names '
                        'of classes in the downloaders module. By default, '
                        'will use cached downloaded data if available, see '
                        '--download-cache-dir and --clear-download-cache. To '
                        'download no data, provide an input of "None"')
    parser.add_argument('--degradations', metavar='deg_name', nargs='*',
                        choices=list(degradations.DEGRADATIONS.keys()),
                        default=list(degradations.DEGRADATIONS.keys()),
                        help='degradations to use on the data. Must match '
                        'names of functions in the degradations module. By '
                        'default, will use them all.')
    parser.add_argument('--excerpt-length', metavar='ms', type=int,
                        help='The length of the excerpt (in ms) to take from '
                        'each piece. The excerpt will start on a note onset '
                        'and include all notes whose onset lies within this '
                        'number of ms after the first note.', default=5000)
    parser.add_argument('--min-notes', metavar='N', type=int, default=10,
                        help='The minimum number of notes required for an '
                        'excerpt to be valid.')
    parser.add_argument('--degradation-kwargs', metavar='json_file_or_string',
                        help='json file or json-formatted string with keyword '
                        'arguments for the degradation functions. First '
                        'provide the degradation name, then a double '
                        'underscore, then the keyword argument name, followed '
                        'by the value to use for the kwarg. e.g. '
                        '`{"time_shift__align_onset": true, '
                        '"pitch_shift__min_pitch": 5}`', default=None)
    parser.add_argument('--degradation-dist', metavar='relative_probability',
                        nargs='*', default=None, help='A list of relative '
                        'probabilities that each degradation will used. Must '
                        'be the same length as --degradations. Defaults to a '
                        'uniform distribution.', type=float)
    parser.add_argument('--clean-prop', type=float, help='The proportion of '
                        'excerpts in the final dataset that should be clean.',
                        default=1 / (1 + len(degradations.DEGRADATIONS)))
    parser.add_argument('--splits', metavar=('train', 'valid', 'test'),
                        nargs=3, type=float, help='The relative sizes of the '
                        'train, validation, and test sets respectively.',
                        default=[0.8, 0.1, 0.1])
    parser.add_argument('--seed', type=int, default=None, help='The numpy seed'
                        ' to use when creating the dataset.')
    parser.add_argument('--clean', action='store_true', help='Clear and delete'
                        f' the download cache {downloaders.DEFAULT_CACHE_PATH}'
                        ' (and do nothing else).')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose '
                        'printing.')
    args = parser.parse_args(args=args_input)
    return args


if __name__ == '__main__':
    ARGS = parse_args()

    if ARGS.clean:
        clean_ok = clean_download_cache(downloaders.DEFAULT_CACHE_PATH)
        sys.exit(0 if clean_ok else 1)

    if ARGS.seed is None:
        seed = np.random.randint(0, 2**32)
        print(f'No random seed supplied. Setting to {seed}.')
    else:
        seed = ARGS.seed
        print(f'Setting random seed to {seed}.')
    np.random.seed(seed)

    # Load given degradation_kwargs
    degradation_kwargs = {}
    if ARGS.degradation_kwargs is not None:
        if os.path.exists(ARGS.degradation_kwargs):
            # If file exists, assume that is what was passed
            with open(ARGS.degradation_kwargs, 'r') as json_file:
                degradation_kwargs = json.load(json_file)
        else:
            # File doesn't exist, assume json string was passed
            degradation_kwargs = json.loads(ARGS.degradation_kwargs)
    degradation_kwargs = parse_degradation_kwargs(degradation_kwargs)
    if ARGS.verbose:
        print(f"Using degradation kwargs: {degradation_kwargs}")

    # Load config
    if ARGS.config is not None:
        with open(ARGS.config, 'r') as file:
            config = json.load(file)
        if ARGS.verbose:
            print(f'Loading from config file {ARGS.config}.')
        if 'degradation_dist' in config:
            ARGS.degradation_dist = np.array(config['degradation_dist'])
            ARGS.degradations = list(degradations.DEGRADATIONS.keys())
        if 'clean_prop' in config:
            ARGS.clean_prop = config['clean_prop']
    # Warn user they specified kwargs for degradation not being used
    for deg, args in degradation_kwargs.items():
        if deg not in ARGS.degradations and len(args) > 0:
            warnings.warn("--degradation_kwargs contains args for unused "
                          f'degradation "{deg}".', UserWarning)

    # Exit if degradation-dist is a diff length to degradations
    if ARGS.degradation_dist is None:
        ARGS.degradation_dist = np.ones(len(ARGS.degradations))
    assert len(ARGS.degradation_dist) == len(ARGS.degradations), (
        "Given degradation_dist is not the same length as degradations:\n"
        f"len({ARGS.degradation_dist}) != len({ARGS.degradations})"
    )

    # Check that no probabilities are invalid
    assert min(ARGS.degradation_dist) >= 0, ("--degradation-dist values must "
                                             "not be negative.")
    assert sum(ARGS.degradation_dist) > 0, ("Some --degradation-dist value "
                                            "must be positive.")
    assert 0 <= ARGS.clean_prop <= 1, ("--clean-prop must be between 0 and 1 "
                                       "(inclusive).")
    assert min(ARGS.splits) >= 0, "--splits values must not be negative."
    assert sum(ARGS.splits) > 0, "Some --splits value must be positive."

    # Parse formats
    if len(ARGS.formats) == 1 and ARGS.formats[0].lower() == 'none':
        formats = []
    else:
        assert all([name in FORMATTERS.keys() for name in ARGS.formats]), (
                f"all provided formats {ARGS.formats} must be in the "
                "list of available formats {list(FORMATTERS.keys())}")
        formats = ARGS.formats

    # These will be tuples of (dataset, relative_path, full_path, note_df).
    # dataset: The name of the dataset the note_df is drawn from. This will be
    #          the excerpt's base directory within output/clean or
    #          output/altered in the generated ACME dataset.
    # relative_path: The relative path of the corresponding file, including
    #                basename, representing the excerpts path within its
    #                dataset base directory.
    # full_path: The full path to the input file. Used for printing errors.
    # note_df: The cleaned note_df read from the input file with the given
    #          input_kwargs.
    # This list will be sorted before shuffling, and the tuples are structured
    # in such a way that the sorting is identical to previous versions of this
    # script to ensure backwards compatability.
    input_data = []
    input_kwargs = {
        'single_track': True,
        'non_overlapping': True
    }


    # Instantiate downloaders =================================================
    OVERWRITE = None
    ds_names = ARGS.datasets
    if len(ds_names) == 1 and ds_names[0].lower() == 'none':
        ds_names = []
    else:
        assert all([name in downloaders.DATASETS for name in ds_names]), (
                f"all provided dataset names {ds_names} must be in the "
                "list of available datasets for download "
                f"{downloaders.DATASETS}")
    # Instantiated downloader classes
    downloader_dict = {
        ds_name: getattr(downloaders, ds_name)(
                     cache_path=downloaders.DEFAULT_CACHE_PATH
                 )
        for ds_name in ds_names
    }


    # Clear and set up output dir =============================================
    if os.path.exists(ARGS.output_dir):
        if ARGS.verbose:
            print(f'Clearing stale data from {ARGS.output_dir}.')
        shutil.rmtree(ARGS.output_dir)
    os.makedirs(ARGS.output_dir, exist_ok=True)
    for out_subdir in ['clean', 'altered']:
        output_dirs = [os.path.join(ARGS.output_dir, out_subdir, name)
                       for name in ds_names]
        for path in output_dirs:
            os.makedirs(path, exist_ok=True)


    # Load data from downloaders ==============================================
    print('Loading data from downloaders, this could take a while...')
    for dataset in downloader_dict:
        downloader = downloader_dict[dataset]

        dataset_base = os.path.join(downloaders.DEFAULT_CACHE_PATH, dataset)
        dataset_base_len = len(dataset_base) + len(os.path.sep)
        output_path = os.path.join(dataset_base, 'data')

        try:
            try:
                downloader.download_csv(output_path=output_path,
                                        overwrite=OVERWRITE,
                                        verbose=ARGS.verbose)
                ext = 'csv'
                input_func = fileio.csv_to_df
            except NotImplementedError:
                downloader.download_midi(output_path=output_path,
                                        overwrite=OVERWRITE,
                                        verbose=ARGS.verbose)
                ext = 'mid'
                input_func = fileio.midi_to_df
        except BadZipfile:
            print("The download cache contains invalid data. Run "
                  "`make_dataset.py --clean` to clean the cache, then try to "
                  "re-create the ACME dataset.", file=sys.stderr)
            sys.exit(1)

        for filename in tqdm(glob(os.path.join(output_path, '**', f'*.{ext}'),
                                  recursive=True),
                             desc=f'Loading data from {dataset}'):

            note_df = input_func(filename, **input_kwargs)
            if note_df is not None:
                rel_path = filename[dataset_base_len + 5:]
                input_data.append((dataset, rel_path, filename, note_df))


    # Load user data ==========================================================
    for data_type in ['midi', 'csv']:
        if data_type == 'midi':
            local_dirs = ARGS.local_midi_dirs
            ext = 'mid'
            df_load_func = fileio.midi_to_df
        else:
            local_dirs = ARGS.local_csv_dirs
            ext = 'csv'
            df_load_func = fileio.csv_to_df

        for path in local_dirs:
            # Bugfix for paths ending in /
            if len(path) > 1 and path[-1] == os.path.sep:
                path = path[:-1]
            dataset = f'local_{os.path.basename(path)}'
            dataset_base_len = len(path) + len(os.path.sep)

            if ARGS.recursive:
                path = os.path.join(path, '**')
            for filepath in tqdm(glob(os.path.join(path, f'*.{ext}'),
                                    recursive=ARGS.recursive),
                                 desc=f'Loading user {data_type} from {path}'):

                note_df = df_load_func(filepath, **input_kwargs)
                if note_df is not None:
                    rel_path = filepath[dataset_base_len:]
                    input_data.append((dataset, rel_path, filepath, note_df))


    # All data is loaded. Sort and shuffle. ===================================
    # output to output_dir/clean/dataset_name/filename.csv
    # The reason for this is we know there will be no filename duplicates
    input_data.sort()
    np.random.shuffle(input_data) # This is important for join_notes

    meta_file = open(os.path.join(ARGS.output_dir, 'metadata.csv'), 'w')


    # Perform degradations and write degraded data to output ==================
    # output to output_dir/degraded/dataset_name/filename.csv
    # The reason for this is that there could be filename duplicates (as above,
    # it's assumed there shouldn't be duplicates within each dataset!), and
    # this allows for easy matching of source and target data
    deg_choices = ARGS.degradations
    goal_deg_dist = ARGS.degradation_dist

    # Remove any degs with goal_deg_dist == 0 and normalize
    non_zero = [i for i, p in enumerate(goal_deg_dist) if p != 0]
    deg_choices = np.array(deg_choices)[non_zero]
    goal_deg_dist = np.array(goal_deg_dist)[non_zero]
    goal_deg_dist /= np.sum(goal_deg_dist)

    # Add none for no degradation
    if 0 < ARGS.clean_prop < 1:
        deg_choices = np.insert(deg_choices, 0, 'none')
        goal_deg_dist *= 1 - ARGS.clean_prop
        goal_deg_dist = np.insert(goal_deg_dist, 0, ARGS.clean_prop)
    elif ARGS.clean_prop == 1:
        deg_choices = np.array(['none'])
        goal_deg_dist = np.array([1])
    nr_degs = len(goal_deg_dist)

    # Normalize split proportions and remove 0s
    split_names = ['train', 'valid', 'test']
    split_props = np.array(ARGS.splits)
    split_props /= np.sum(split_props)
    non_zero = [i for i, p in enumerate(split_props) if p != 0]
    split_names = np.array(split_names)[non_zero]
    split_props = np.array(split_props)[non_zero]
    nr_splits = len(split_names)

    # Write out deg_choices to degradation_ids.csv
    with open(os.path.join(ARGS.output_dir, 'degradation_ids.csv'),
              'w') as file:
        file.write('id,degradation_name\n')
        for i, deg_name in enumerate(deg_choices):
            file.write(f'{i},{deg_name}\n')

    # The idea is to keep track of the current distribution of degradations
    # and then sample in reverse order of the difference between this and
    # the goal distribution. We do the same for splits.
    deg_counts = np.zeros(nr_degs)
    split_counts = np.zeros(nr_splits)

    meta_file.write('altered_csv_path,degraded,degradation_id,'
                    'clean_csv_path,split\n')
    for i, data in enumerate(tqdm(input_data, desc="Degrading data")):
        dataset, rel_path, file_path, note_df = data
        rel_path = f'{rel_path[:-3]}csv'
        # First, get the degradation order for this iteration.
        # Get the current distribution of degradations
        if np.sum(deg_counts) == 0: # First iteration, set to uniform
            current_deg_dist = np.ones(nr_degs) / nr_degs
            current_split_dist = np.ones(nr_splits) / nr_splits
        else:
            current_deg_dist = deg_counts / np.sum(deg_counts)
            current_split_dist = split_counts / np.sum(split_counts)

        # Grab an excerpt from this df
        excerpt = get_random_excerpt(note_df, min_notes=ARGS.min_notes,
                                     excerpt_length=ARGS.excerpt_length,
                                     first_onset_range=(0, 200), iterations=10)

        # If no valid excerpt was found, skip this piece
        if excerpt is None:
            warnings.warn("Unable to find valid excerpt from file "
                          f"{file_path}. Lengthen --excerpt-length or "
                          "lower --min-notes. Skipping.", UserWarning)
            continue

        # Try degradations in reverse order of the difference between
        # their current distribution and their desired distribution.
        diffs = goal_deg_dist - current_deg_dist
        degs_sorted = sorted(zip(diffs, deg_choices,
                                 list(range(len(deg_choices)))))[::-1]

        # Calculate split in the same way (but only save the first)
        split_diffs = split_props - current_split_dist
        _, split_name, split_num = sorted(
            zip(
                    split_diffs,
                    split_names,
                    list(range(nr_splits))
            )
        )[-1]

        # Make default labels for no degradation
        clean_path = os.path.join('clean', dataset, rel_path)
        altered_path = clean_path
        deg_binary = 0

        # Try to perform a degradation
        degraded = None
        for diff, deg_name, deg_num in degs_sorted:
            # Break for no degradation
            if deg_name == 'none':
                break

            # Try the degradation
            deg_fun = degradations.DEGRADATIONS[deg_name]
            deg_fun_kwargs = degradation_kwargs[deg_name] # degradation_kwargs
                                                          # at top of main call
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                degraded = deg_fun(excerpt, **deg_fun_kwargs)

            if degraded is not None:
                # Update labels
                deg_binary = 1
                altered_path = os.path.join('altered', dataset, rel_path)

                # Write degraded csv
                altered_outpath = os.path.join(ARGS.output_dir, altered_path)
                fileio.df_to_csv(degraded, altered_outpath)
                break

        # Write data
        if not (degraded is None and ARGS.clean_prop == 0):
            # Update counts
            deg_counts[deg_num] += 1
            split_counts[split_num] += 1

            # Write clean csv
            clean_outpath = os.path.join(ARGS.output_dir, clean_path)
            fileio.df_to_csv(excerpt, clean_outpath)

            # Write metadata
            meta_file.write(f'{altered_path},{deg_binary},{deg_num},'
                            f'{clean_path},{split_name}\n')
        else:
            warnings.warn("Unable to degrade chosen excerpt from "
                          f"{file_path} and no clean excerpts requested."
                          " Skipping.", UserWarning)

    meta_file.close()

    for f in formats:
        create_corpus_csvs(ARGS.output_dir, FORMATTERS[f])

    print(f'\n{10*"="} Finished! {10*"="}\n')
    print(f'Count of degradations:')
    for deg_name, count in zip(deg_choices, deg_counts):
        print(f'\t* {deg_name}: {int(count)}')

    print(f'\nYou will find the generated data at {ARGS.output_dir} '
          'with subdirectories')
    print(f'\t* clean - contains the extracted clean excerpts')
    print(f'\t* altered - contains the excerpts altered by the degradations '
          'described in metadata.csv')
    print('\nmetadata.csv describes:')
    print('\t* (the id number for) the type of degradation used for the '
          'alteration')
    print('\t* the path for the altered and clean files')
    print('\t* which split (train, valid, test) the file should be used in')
    print('\t* in which corpus and on what line the file is located')

    print('\ndegradation_ids.csv is a mapping of degradation name to the id '
          'number used in metadata.csv')

    for f in formats:
        print(f"\n{FORMATTERS[f]['message']}")

    print('\nTo reproduce this dataset again, run the script with argument '
          f'--seed {seed}')
    print(LOGO)
