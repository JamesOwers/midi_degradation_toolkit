"""Quick utils for files and plotting used by the adjacent notebooks."""
import logging
import zipfile
from pathlib import Path

import requests

LOGGER = logging.getLogger(__name__)


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
        zz.extractall(path=stem)
