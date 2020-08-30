"""Utility functions for file manipulation"""
import logging
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile


def download_file(source, dest, verbose=False, overwrite=None):
    """Get a file from a url and save it locally"""
    if verbose:
        print(f"Downloading {source} to {dest}")
    if os.path.exists(dest):
        if overwrite is None:
            if verbose:
                logging.warning(f"WARNING: {dest} already exists, not downloading")
            return
        if not overwrite:
            raise OSError(f"{dest} already exists")
    try:
        urllib.request.urlretrieve(source, dest)
    except urllib.error.HTTPError as e:
        print(f"Url {source} does not exist", file=sys.stderr)
        raise e


def make_directory(path, overwrite=None, verbose=False):
    """Convenience function to create a directory and handle cases where
        it already exists.

        Args
        ----
        path: str
            The path of the directory to create
        overwrite: boolean or None
            If the path already exists, if overwrite is: True - delete the
            existing path; False - return error; None - leave the existing
            path as it is and throw a warning
        verbose: bool
            Verbosity of printing
        """
    if verbose:
        print(f"Making directory at {path}")

    mkdir = os.makedirs
    try:
        mkdir(path)
    except FileExistsError as e:
        if overwrite is True:
            if verbose:
                print(f"Deleting existing directory: {path}")
            shutil.rmtree(path)
            mkdir(path)
        elif overwrite is None:
            if verbose:
                logging.warning(
                    f"WARNING: {path} already exists, writing "
                    "files only if they do not already exist.",
                )
        elif overwrite is False:
            raise e
        else:
            raise ValueError(
                "overwrite should be boolean or None, not " f'"{overwrite}"'
            )


def extract_zip(zip_path, out_path, overwrite=None, verbose=False):
    """Convenience function to extract zip file to out_path."""
    if verbose:
        print(f"Extracting {zip_path} to {out_path}")
    dirname = os.path.splitext(os.path.basename(zip_path))[0]
    extracted_path = os.path.join(out_path, dirname)
    if os.path.exists(extracted_path):
        if overwrite is True:
            if verbose:
                logging.warning("Deleting existing directory: " f"{extracted_path}")
            shutil.rmtree(extracted_path)
        elif overwrite is None:
            if verbose:
                logging.warning(
                    f"{extracted_path} already exists. Assuming "
                    "this zip has already been extracted, not "
                    "extracting.",
                )
            return extracted_path
        elif overwrite is False:
            raise FileExistsError(f"{extracted_path} already exists")

    with zipfile.ZipFile(zip_path, "r") as zz:
        zz.extractall(path=out_path)
    return extracted_path


def copy_file(filepath, output_path, overwrite=None, mkdir=False):
    """Convenience function to copy a file from filepath to output_path."""
    path = os.path.join(output_path, os.path.basename(filepath))
    if os.path.exists(path):
        if overwrite is True:
            shutil.copy(filepath, output_path)
        elif overwrite is None:
            return
        elif overwrite is False:
            raise FileExistsError(f"{path} already exists")
    else:
        shutil.copy(filepath, output_path)
