# -*- coding: utf-8 -*-
"""Download file."""
import hashlib
import os
import shutil
import sys
import tarfile
import time
import typing
import zipfile
from pathlib import Path

import numpy as np
import six
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError


class Progbar(object):
    """
    Displays a progress bar.

    :param target: Total number of steps expected, None if unknown.
    :param width: Progress bar width on screen.
    :param verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
    :param stateful_metrics: Iterable of string names of metrics that
        should *not* be averaged over time. Metrics in this list
        will be displayed as-is. All others will be averaged
        by the progbar before display.
    :param interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(
            self,
            target,
            width=30,
            verbose=1,
            interval=0.05,
    ):
        """Init."""
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval

        self._dynamic_display = ((hasattr(sys.stdout,
                                          'isatty') and sys.stdout.isatty()
                                  ) or 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._start = time.time()
        self._last_update = 0

    def update(self, current):
        """Updates the progress bar."""
        self._seen_so_far = current

        now = time.time()
        info = ' - {0:.0f}s'.format(now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and self.target is not
                None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                bar = '{2:{0:d}d}/{1} ['.format(
                    numdigits, self.target, current)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '{0:7d}/Unknown'.format(current)

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = int(time_per_unit * (self.target - current))
                if eta > 3600:
                    eta_format = ('{0:d}:{1:02d}:{2:02d}'.format(
                        eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '{0:d}:{1:02d}'.format(eta // 60, eta % 60)
                else:
                    eta_format = '{0:d}s'.format(eta)

                info = ' - ETA: {0}'.format(eta_format)
            else:
                if time_per_unit >= 1:
                    info += ' {0:.0f}s/step'.format(time_per_unit)
                elif time_per_unit >= 1e-3:
                    info += ' {0:.0f}ms/step'.format(time_per_unit * 1e3)
                else:
                    info += ' {0:.0f}us/step'.format(time_per_unit * 1e6)

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                info += '\n'
                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now


def _extract_archive(file_path, path='.', archive_format='auto'):
    """
    Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    :param file_path: path to the archive file
    :param path: path to extract the archive file
    :param archive_format: Archive format to try for extracting the file.
        Options are 'auto', 'tar', 'zip', and None.
        'tar' includes tar, tar.gz, and tar.bz files.
        The default 'auto' is ['tar', 'zip'].
        None or an empty list will return no matches found.

    :return: True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError,
                        KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def get_file(
        fname: str = None,
        origin: str = None,
        untar: bool = False,
        extract: bool = False,
        md5_hash: typing.Any = None,
        file_hash: typing.Any = None,
        hash_algorithm: str = 'auto',
        archive_format: str = 'auto',
        cache_subdir: typing.Union[Path, str] = 'data',
        cache_dir: typing.Union[Path, str] = 'dataset',
        verbose: int = 1
) -> str:
    """
    Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.project/datasets`, placed in the cache_subdir `data`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.project/datasets/data/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.

    :param fname: Name of the file. If an absolute path `/path/to/file.txt` is
        specified the file will be saved at that location.
    :param origin: Original URL of the file.
    :param untar: Deprecated in favor of 'extract'. Boolean, whether the file
        should be decompressed.
    :param md5_hash: Deprecated in favor of 'file_hash'. md5 hash of the file
        for verification.
    :param file_hash: The expected hash string of the file after download.
        The sha256 and md5 hash algorithms are both supported.
    :param cache_subdir: Subdirectory under the cache dir where the file is
        saved. If an absolute path `/path/to/folder` is specified the file
        will be saved at that location.
    :param hash_algorithm: Select the hash algorithm to verify the file.
        options are 'md5', 'sha256', and 'auto'. The default 'auto' detects
        the hash algorithm in use.
    :papram extract: True tries extracting the file as an Archive, like tar
        or zip.
    :param archive_format: Archive format to try for extracting the file.
        Options are 'auto', 'tar', 'zip', and None.
        'tar' includes tar, tar.gz, and tar.bz files.
        The default 'auto' is ['tar', 'zip'].
        None or an empty list will return no matches found.
    :param cache_dir: Location to store cached files, when None it defaults to
        the [project.USER_DATA_DIR](~/.project/datasets).
    :param verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)

    :return: Path to the downloaded file.
    """
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.simtext')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the file hash '
                      'does not match the original value of file_hash.'
                      ' We will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        class ProgressTracker(object):
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(
                    target=total_size, verbose=verbose)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {} : {} -- {}'
        try:
            try:
                from six.moves.urllib.request import urlretrieve
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        ProgressTracker.progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """
    Validates a file against a sha256 or md5 hash.

    :param fpath: path to the file being validated
    :param file_hash:  The expected hash string of the file.
        The sha256 and md5 hash algorithms are both supported.
    :param algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
        The default 'auto' detects the hash algorithm in use.
    :param chunk_size: Bytes to read at a time, important for large files.

    :return: Whether the file is valid.
    """
    if ((algorithm == 'sha256') or (algorithm == 'auto' and len(
            file_hash) == 64)):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """
    Calculates a file sha256 or md5 hash.

    :param fpath: path to the file being validated
    :param algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
        The default 'auto' detects the hash algorithm in use.
    :param chunk_size: Bytes to read at a time, important for large files.

    :return: The file hash.
    """
    if algorithm == 'sha256':
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()
